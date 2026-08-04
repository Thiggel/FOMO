"""Fixed-DINOv3 geometry and generated-image diagnostic report.

Each input is one treatment group and may be a glob of ordinary image files or
BRIDGE's HDF5 image stores.  The DINOv3 encoder is fitted once and never
updated; this prevents apparent density changes from being caused solely by
the evolving SSL encoder's scale.
"""
import argparse
import glob
import io
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from transformers import AutoImageProcessor, AutoModel


def load_group(spec, limit):
    images, names = [], []
    for path in glob.glob(spec, recursive=True):
        if len(images) >= limit:
            break
        if path.endswith(".h5"):
            with h5py.File(path, "r") as handle:
                for idx, encoded in enumerate(handle["images"]):
                    if len(images) >= limit:
                        break
                    images.append(Image.open(io.BytesIO(bytes(encoded))).convert("RGB"))
                    names.append(f"{path}:{idx}")
        elif path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            images.append(Image.open(path).convert("RGB"))
            names.append(path)
    return images, names


def embed(images, processor, model, device):
    outputs = []
    for start in range(0, len(images), 32):
        batch = processor(images=images[start : start + 32], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs.append(model(**batch).last_hidden_state[:, 0].cpu())
    return torch.nn.functional.normalize(torch.cat(outputs), dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()
    images, names, groups = [], [], []
    for group, spec in enumerate(args.inputs):
        group_images, group_names = load_group(spec, args.max_images)
        images.extend(group_images)
        names.extend(group_names)
        groups.extend([group] * len(group_images))
    if len(images) < 3:
        raise RuntimeError("Need at least three generated images for fixed-space report")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    load_kwargs = {
        "cache_dir": os.environ.get("HF_HUB_CACHE"),
        "local_files_only": os.environ.get("HF_HUB_OFFLINE") == "1",
    }
    processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
    model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
    features = embed(images, processor, model, device)
    neighbors = NearestNeighbors(n_neighbors=min(6, len(features)), metric="cosine").fit(features)
    distances, _ = neighbors.kneighbors(features)
    radii = distances[:, -1]
    groups = np.asarray(groups)
    report = {"n_images": int(len(features)), "groups": {}}
    for group in sorted(set(groups)):
        selected = groups == group
        rgb = np.concatenate([np.asarray(image.resize((32, 32))).reshape(-1, 3) for image, flag in zip(images, selected) if flag])
        report["groups"][str(group)] = {
            "n_images": int(selected.sum()),
            "median_radius": float(np.median(radii[selected])),
            "p90_radius": float(np.quantile(radii[selected], 0.9)),
            "near_duplicate_rate": float((distances[selected, 1] < 0.02).mean()),
            # A reproducible automated artifact proxy, reported separately
            # from semantic fidelity rather than presented as human judgment.
            "pixel_clipping_rate": float(((rgb <= 2) | (rgb >= 253)).mean()),
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "dinov3_embeddings.npz", features=features, groups=groups, radii=radii, paths=np.asarray(names))
    with open(output / "metrics.json", "w") as handle:
        json.dump(report, handle, indent=2)
    z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(2, (len(features) - 1) // 3)), random_state=0).fit_transform(features)
    plt.figure(figsize=(6, 5))
    plt.scatter(z[:, 0], z[:, 1], c=groups, s=4, cmap="tab10")
    plt.tight_layout()
    plt.savefig(output / "joint_tsne.png", dpi=220)


if __name__ == "__main__":
    main()
