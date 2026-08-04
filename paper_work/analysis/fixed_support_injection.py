"""Measure where generated samples add support in a fixed pre-repair space.

This analysis deliberately freezes the cycle-0 encoder.  It therefore tests a
data intervention claim, not whether subsequent SSL training changes the
coordinate system preferentially around treated anchors.
"""

import argparse
import io
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.models.backbones.Resnet import ResNet50
from experiment.utils.set_seed import set_seed


def load_encoder(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model = ResNet50(128)
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")},
        strict=False,
    )
    return model.to(device).eval()


@torch.inference_mode()
def embed(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features, labels = [], []
    for images, targets in loader:
        features.append(model.extract_features(images.to(device)).float().cpu())
        labels.append(torch.as_tensor(targets).cpu())
    return (
        F.normalize(torch.cat(features), dim=1).numpy().astype("float32"),
        torch.cat(labels).numpy().astype("int64"),
    )


class GeneratedImages(Dataset):
    def __init__(self, run_root, manifests, transform):
        self.items = []
        self.transform = transform
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text())
            cycle = int(payload["cycle"])
            for row in payload["rows"]:
                repair_index = int(row["repair_index"])
                file_index, local_index = divmod(repair_index, 1000)
                path = run_root / "generated" / str(cycle) / f"images_{file_index}.h5"
                self.items.append((path, local_index, int(row["anchor_label"]), cycle))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, local_index, label, _ = self.items[index]
        with h5py.File(path, "r") as handle:
            image = Image.open(io.BytesIO(bytes(handle["images"][local_index]))).convert("RGB")
        return self.transform(image), label


def knn_metrics(query, reference, query_labels, reference_labels, k, self_reference=False):
    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(reference)
    offset = 1 if self_reference else 0
    similarities, neighbors = index.search(query, min(k + offset, len(reference)))
    if self_reference:
        similarities, neighbors = similarities[:, 1:], neighbors[:, 1:]
    radius = 1.0 - similarities[:, :k].mean(axis=1)
    purity = (reference_labels[neighbors[:, :k]] == query_labels[:, None]).mean(axis=1)
    return radius, purity


def match_controls(anchor_rows, eligible_rows, radii, labels):
    unused = set(int(row) for row in eligible_rows)
    anchors, controls = [], []
    for anchor in sorted(anchor_rows, key=lambda row: radii[row]):
        pool = [row for row in unused if labels[row] == labels[anchor]] or list(unused)
        if not pool:
            break
        control = min(pool, key=lambda row: abs(float(radii[row] - radii[anchor])))
        anchors.append(anchor)
        controls.append(control)
        unused.remove(control)
    return np.asarray(anchors), np.asarray(controls)


def summarize(rows, before_radius, after_radius, before_purity, after_purity):
    return {
        "n": int(len(rows)),
        "radius_before": float(before_radius[rows].mean()),
        "radius_after": float(after_radius[rows].mean()),
        "radius_change": float((after_radius[rows] - before_radius[rows]).mean()),
        "purity_before": float(before_purity[rows].mean()),
        "purity_after": float(after_purity[rows].mean()),
        "purity_change": float((after_purity[rows] - before_purity[rows]).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dm = ImbalancedDataModule(
        dataset_path="clane9/imagenet-100",
        imbalance_method=ImbalanceMethods.PowerLawImbalance,
        transform=transform,
        additional_data_path=f"/tmp/fomo_fixed_support_seed{args.seed}",
    )
    train = dm.train_dataset
    underlying = np.asarray(train.indices if isinstance(train, Subset) else np.arange(len(train)))
    row_for_underlying = {int(value): row for row, value in enumerate(underlying)}
    run_root = Path(args.run_root)
    manifests = sorted((run_root / "generated" / "repair_manifests").glob("cycle_*.json"))
    if not manifests:
        raise RuntimeError("No repair manifests found")
    selected_per_cycle, ever_selected = [], set()
    for path in manifests:
        payload = json.loads(path.read_text())
        selected = {int(row["anchor_underlying_index"]) for row in payload["rows"]}
        selected_per_cycle.append(selected)
        ever_selected.update(selected)
    anchor_rows = np.asarray([row_for_underlying[index] for index in selected_per_cycle[0] if index in row_for_underlying])
    eligible = np.asarray([row for row, value in enumerate(underlying) if int(value) not in ever_selected])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_encoder(args.base, device)
    original_features, original_labels = embed(model, train, device)
    before_radius, before_purity = knn_metrics(
        original_features, original_features, original_labels, original_labels, args.k, self_reference=True
    )
    anchor_rows, control_rows = match_controls(anchor_rows, eligible, before_radius, original_labels)

    generated = GeneratedImages(run_root, manifests, transform)
    generated_features, generated_labels = embed(model, generated, device, batch_size=128)
    augmented_features = np.concatenate([original_features, generated_features])
    augmented_labels = np.concatenate([original_labels, generated_labels])
    # The first neighbor is the query image itself, which is present in the original reference block.
    index = faiss.IndexFlatIP(augmented_features.shape[1])
    index.add(augmented_features)
    similarities, neighbors = index.search(original_features, args.k + 1)
    after_radius = 1.0 - similarities[:, 1:].mean(axis=1)
    after_purity = (augmented_labels[neighbors[:, 1:]] == original_labels[:, None]).mean(axis=1)

    anchors = summarize(anchor_rows, before_radius, after_radius, before_purity, after_purity)
    controls = summarize(control_rows, before_radius, after_radius, before_purity, after_purity)
    report = {
        "seed": args.seed,
        "k": args.k,
        "claim": "support added by generated samples in the frozen pre-repair representation",
        "n_generated": len(generated),
        "anchors": anchors,
        "matched_controls": controls,
        "difference_in_differences": {
            "radius_change": anchors["radius_change"] - controls["radius_change"],
            "purity_change": anchors["purity_change"] - controls["purity_change"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
