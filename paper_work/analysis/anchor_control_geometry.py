"""Compare local geometry at BRIDGE anchors and radius-matched controls.

The matching variables and labels come from the common pre-repair checkpoint.
Labels are used only for post-hoc matching and neighborhood-purity analysis;
they are never exposed to BRIDGE during training or acquisition.
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.models.backbones.Resnet import ResNet50
from experiment.utils.set_seed import set_seed


def load_encoder(path: str, device: torch.device) -> ResNet50:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model = ResNet50(128)
    model.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        },
        strict=False,
    )
    return model.to(device).eval()


@torch.inference_mode()
def embed(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    features, labels = [], []
    for images, targets in loader:
        features.append(model.extract_features(images.to(device)).float().cpu())
        labels.append(torch.as_tensor(targets).cpu())
    features = F.normalize(torch.cat(features), dim=1).numpy().astype("float32")
    return features, torch.cat(labels).numpy().astype("int64")


def local_metrics(features, labels, k):
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    similarities, neighbors = index.search(features, min(k + 1, len(features)))
    similarities = similarities[:, 1:]
    neighbors = neighbors[:, 1:]
    radius = 1.0 - similarities.mean(axis=1)
    purity = (labels[neighbors] == labels[:, None]).mean(axis=1)
    return radius, purity


def match_controls(anchor_rows, eligible_rows, radii, labels):
    """Greedy 1:1 matching on class and pre-repair local radius."""
    unused = set(int(row) for row in eligible_rows)
    controls = []
    order = sorted(anchor_rows, key=lambda row: radii[row])
    for anchor in order:
        same_class = [row for row in unused if labels[row] == labels[anchor]]
        pool = same_class if same_class else list(unused)
        if not pool:
            break
        control = min(pool, key=lambda row: abs(float(radii[row] - radii[anchor])))
        controls.append(control)
        unused.remove(control)
    return np.asarray(order[: len(controls)], dtype=np.int64), np.asarray(controls, dtype=np.int64)


def cohort_summary(rows, before_radius, after_radius, before_purity, after_purity):
    radius_delta = after_radius[rows] - before_radius[rows]
    purity_delta = after_purity[rows] - before_purity[rows]
    return {
        "n": int(len(rows)),
        "radius_before": float(np.mean(before_radius[rows])),
        "radius_after": float(np.mean(after_radius[rows])),
        "radius_change": float(np.mean(radius_delta)),
        "radius_change_median": float(np.median(radius_delta)),
        "purity_before": float(np.mean(before_purity[rows])),
        "purity_after": float(np.mean(after_purity[rows])),
        "purity_change": float(np.mean(purity_delta)),
        "purity_change_median": float(np.median(purity_delta)),
    }


def evaluate_transition(before, after, labels, anchor_rows, control_rows, k):
    before_radius, before_purity = local_metrics(before, labels, k)
    after_radius, after_purity = local_metrics(after, labels, k)
    anchors = cohort_summary(
        anchor_rows, before_radius, after_radius, before_purity, after_purity
    )
    controls = cohort_summary(
        control_rows, before_radius, after_radius, before_purity, after_purity
    )
    return {
        "anchors": anchors,
        "matched_controls": controls,
        "difference_in_differences": {
            "radius_change": anchors["radius_change"] - controls["radius_change"],
            "purity_change": anchors["purity_change"] - controls["purity_change"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--no-repair")
    parser.add_argument("--selections", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    set_seed(args.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    dm = ImbalancedDataModule(
        dataset_path="clane9/imagenet-100",
        imbalance_method=ImbalanceMethods.PowerLawImbalance,
        transform=transform,
        additional_data_path=f"/tmp/fomo_anchor_control_seed{args.seed}",
    )
    train = dm.train_dataset
    if isinstance(train, Subset):
        underlying = np.asarray(train.indices, dtype=np.int64)
    else:
        underlying = np.arange(len(train), dtype=np.int64)
    row_for_underlying = {int(index): row for row, index in enumerate(underlying)}

    selections = [np.load(path) for path in args.selections]
    treated_underlying = set(
        int(index) for index in selections[0]["selected_underlying_indices"]
    )
    ever_treated = set()
    for selection in selections:
        ever_treated.update(
            int(index) for index in selection["selected_underlying_indices"]
        )
    anchor_rows = np.asarray(
        [row_for_underlying[index] for index in treated_underlying if index in row_for_underlying],
        dtype=np.int64,
    )
    eligible_rows = np.asarray(
        [row for row, index in enumerate(underlying) if int(index) not in ever_treated],
        dtype=np.int64,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_encoder(args.base, device)
    base_features, labels = embed(base_model, train, device)
    del base_model
    torch.cuda.empty_cache()
    base_radius, _ = local_metrics(base_features, labels, args.k)
    anchor_rows, control_rows = match_controls(
        anchor_rows, eligible_rows, base_radius, labels
    )

    bridge_model = load_encoder(args.bridge, device)
    bridge_features, bridge_labels = embed(bridge_model, train, device)
    del bridge_model
    torch.cuda.empty_cache()
    if not np.array_equal(labels, bridge_labels):
        raise RuntimeError("Dataset order differs between the paired checkpoints")

    report = {
        "seed": args.seed,
        "k": args.k,
        "matching": "one-to-one within-class nearest initial kNN radius; controls never selected in any repair cycle",
        "label_use": "post-hoc analysis only",
        "bridge": evaluate_transition(
            base_features,
            bridge_features,
            labels,
            anchor_rows,
            control_rows,
            args.k,
        ),
        "matched_anchor_underlying_indices": underlying[anchor_rows].tolist(),
        "matched_control_underlying_indices": underlying[control_rows].tolist(),
    }
    if args.no_repair:
        no_repair_model = load_encoder(args.no_repair, device)
        no_repair_features, no_repair_labels = embed(no_repair_model, train, device)
        if not np.array_equal(labels, no_repair_labels):
            raise RuntimeError("Dataset order differs in no-repair checkpoint")
        report["continued_training"] = evaluate_transition(
            base_features,
            no_repair_features,
            labels,
            anchor_rows,
            control_rows,
            args.k,
        )
        report["triple_difference"] = {
            metric: report["bridge"]["difference_in_differences"][metric]
            - report["continued_training"]["difference_in_differences"][metric]
            for metric in ("radius_change", "purity_change")
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"matched_anchor_underlying_indices", "matched_control_underlying_indices"}}, indent=2))


if __name__ == "__main__":
    main()
