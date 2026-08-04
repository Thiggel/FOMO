"""Relate class-level representation densification to held-out accuracy gains."""

import argparse
import json
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
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
        {key.removeprefix("model."): value for key, value in state.items()
         if key.startswith("model.")},
        strict=False,
    )
    return model.to(device).eval()


@torch.inference_mode()
def embed(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    features, labels = [], []
    for images, target in loader:
        features.append(model.extract_features(images.to(device)).float().cpu())
        labels.append(torch.as_tensor(target).cpu())
    return F.normalize(torch.cat(features), dim=1).numpy().astype("float32"), torch.cat(labels).numpy()


def class_metrics(train_x, train_y, test_x, test_y, radius_k=20, vote_k=20):
    index = faiss.IndexFlatIP(train_x.shape[1])
    index.add(train_x)
    train_similarity, _ = index.search(train_x, min(radius_k + 1, len(train_x)))
    radii = 1.0 - train_similarity[:, 1:].mean(axis=1)
    _, neighbors = index.search(test_x, min(vote_k, len(train_x)))
    predictions = np.asarray(
        [np.bincount(train_y[row], minlength=100).argmax() for row in neighbors]
    )
    rows = {}
    for class_id in sorted(set(train_y) & set(test_y)):
        train_mask = train_y == class_id
        test_mask = test_y == class_id
        if train_mask.sum() < 3 or test_mask.sum() < 1:
            continue
        rows[int(class_id)] = {
            "train_count": int(train_mask.sum()),
            "median_radius": float(np.median(radii[train_mask])),
            "heldout_accuracy": float((predictions[test_mask] == test_y[test_mask]).mean()),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dm = ImbalancedDataModule(
        dataset_path="clane9/imagenet-100",
        imbalance_method=ImbalanceMethods.PowerLawImbalance,
        transform=transform,
        additional_data_path=f"/tmp/fomo_geometry_seed{args.seed}",
    )
    train = dm.train_dataset
    heldout = ConcatDataset([dm.val_dataset, dm.test_dataset])
    generator = torch.Generator().manual_seed(args.seed)
    if args.max_train and len(train) > args.max_train:
        train = Subset(
            train,
            torch.randperm(len(train), generator=generator)[: args.max_train].tolist(),
        )
    if args.max_test and len(heldout) > args.max_test:
        heldout = Subset(
            heldout,
            torch.randperm(len(heldout), generator=generator)[: args.max_test].tolist(),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_encoder(args.base, device)
    base_train_x, train_y = embed(base_model, train, device)
    base_test_x, test_y = embed(base_model, heldout, device)
    del base_model
    torch.cuda.empty_cache()
    bridge_model = load_encoder(args.bridge, device)
    bridge_train_x, bridge_train_y = embed(bridge_model, train, device)
    bridge_test_x, bridge_test_y = embed(bridge_model, heldout, device)
    assert np.array_equal(train_y, bridge_train_y)
    assert np.array_equal(test_y, bridge_test_y)
    base = class_metrics(base_train_x, train_y, base_test_x, test_y)
    bridge = class_metrics(bridge_train_x, train_y, bridge_test_x, test_y)
    rows = []
    for class_id in sorted(set(base) & set(bridge)):
        rows.append({
            "class_id": class_id,
            "train_count": base[class_id]["train_count"],
            "support_improvement": (
                base[class_id]["median_radius"] - bridge[class_id]["median_radius"]
            ),
            "accuracy_improvement": (
                bridge[class_id]["heldout_accuracy"] - base[class_id]["heldout_accuracy"]
            ),
            "base_radius": base[class_id]["median_radius"],
            "bridge_radius": bridge[class_id]["median_radius"],
            "base_accuracy": base[class_id]["heldout_accuracy"],
            "bridge_accuracy": bridge[class_id]["heldout_accuracy"],
        })
    support = np.asarray([row["support_improvement"] for row in rows])
    accuracy = np.asarray([row["accuracy_improvement"] for row in rows])
    correlation = spearmanr(support, accuracy)
    report = {
        "seed": args.seed,
        "n_classes": len(rows),
        "spearman_r": float(correlation.statistic),
        "spearman_p": float(correlation.pvalue),
        "mean_support_improvement": float(support.mean()),
        "mean_accuracy_improvement": float(accuracy.mean()),
        "classes": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    plt.figure(figsize=(5, 4))
    plt.scatter(support, accuracy, c=np.log1p([row["train_count"] for row in rows]), s=16)
    plt.axhline(0, color="grey", linewidth=0.7)
    plt.axvline(0, color="grey", linewidth=0.7)
    plt.xlabel("local-support improvement (radius decrease)")
    plt.ylabel("held-out class-accuracy improvement")
    plt.title(f"seed {args.seed}: Spearman r={correlation.statistic:.2f}")
    plt.tight_layout()
    plt.savefig(output.with_suffix(".png"), dpi=220)


if __name__ == "__main__":
    main()
