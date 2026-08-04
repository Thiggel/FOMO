"""Fixed-DINOv3 anchor→repair fidelity/diversity report from provenance manifests.

This deliberately reports automatic representation diagnostics, not a claimed
human artifact rate.  Each generated variant is paired with its saved source
anchor, making the semantic-fidelity and diversity measurements auditable.
"""
import argparse
import glob
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from transformers import AutoImageProcessor, AutoModel


def h5_image(run_root: Path, cycle: int, repair_index: int):
    file_idx, local_idx = divmod(repair_index, 1000)
    path = run_root / "generated" / str(cycle) / f"images_{file_idx}.h5"
    with h5py.File(path, "r") as handle:
        return Image.open(io.BytesIO(bytes(handle["images"][local_idx]))).convert("RGB")


def encode(images, processor, model, device):
    outputs = []
    for start in range(0, len(images), 32):
        batch = processor(images=images[start : start + 32], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs.append(model(**batch).last_hidden_state[:, 0].cpu())
    return torch.nn.functional.normalize(torch.cat(outputs), dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", required=True, help="recursive cycle_*.json glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-anchors", type=int, default=200)
    args = parser.parse_args()
    manifests = [Path(item) for item in glob.glob(args.manifests, recursive=True)]
    manifests = [item for item in manifests if "repair_manifests" in str(item)]
    if not manifests:
        raise RuntimeError("No repair provenance manifests found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    load_kwargs = {
        "cache_dir": os.environ.get("HF_HUB_CACHE"),
        "local_files_only": os.environ.get("HF_HUB_OFFLINE") == "1",
    }
    processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
    model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
    summary, viz_features, viz_groups, viz_kinds = [], [], [], []

    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        if payload.get("repair_operator") in {"anchor_duplicate", "oracle_real_restoration"}:
            continue
        rows_by_anchor = defaultdict(list)
        for row in payload["rows"]:
            rows_by_anchor[row["anchor_dataset_position"]].append(row)
        anchors = list(rows_by_anchor.items())[: args.max_anchors]
        run_root = manifest_path.parents[2]
        images, owner, kinds = [], [], []
        for anchor_order, (_, rows) in enumerate(anchors):
            anchor_path = rows[0].get("anchor_image")
            if not anchor_path or not Path(anchor_path).exists():
                continue
            images.append(Image.open(anchor_path).convert("RGB")); owner.append(anchor_order); kinds.append("anchor")
            for row in rows:
                try:
                    images.append(h5_image(run_root, int(payload["cycle"]), int(row["repair_index"])))
                    owner.append(anchor_order); kinds.append("generated")
                except (FileNotFoundError, IndexError, OSError):
                    continue
        if len(images) < 3:
            continue
        features = encode(images, processor, model, device)
        per_anchor_fidelity, per_anchor_diversity = [], []
        for anchor_id in sorted(set(owner)):
            positions = [i for i, value in enumerate(owner) if value == anchor_id]
            anchor_pos = next(i for i in positions if kinds[i] == "anchor")
            generated = [i for i in positions if kinds[i] == "generated"]
            if not generated:
                continue
            per_anchor_fidelity.extend(features[generated] @ features[anchor_pos])
            if len(generated) > 1:
                similarities = features[generated] @ features[generated].T
                per_anchor_diversity.append(float(1 - similarities[np.triu_indices(len(generated), 1)].mean()))
        pixels = np.concatenate([np.asarray(image.resize((32, 32))).reshape(-1, 3) for image, kind in zip(images, kinds) if kind == "generated"])
        label = "/".join(run_root.parts[-3:])
        summary.append({
            "run": label, "cycle": int(payload["cycle"]), "repair_operator": payload["repair_operator"],
            "n_anchors": int(len(set(owner))), "n_generated": int(sum(kind == "generated" for kind in kinds)),
            "dino_anchor_generation_cosine_mean": float(np.mean(per_anchor_fidelity)),
            "dino_anchor_generation_cosine_p10": float(np.quantile(per_anchor_fidelity, .1)),
            "dino_generation_diversity_mean": float(np.mean(per_anchor_diversity)) if per_anchor_diversity else 0.0,
            "pixel_clipping_proxy": float(((pixels <= 2) | (pixels >= 253)).mean()),
        })
        # Keep a bounded joint projection for the appendix figure.
        take = min(len(features), 384)
        viz_features.append(features[:take]); viz_groups += [label] * take; viz_kinds += kinds[:take]

    if not summary:
        raise RuntimeError("No usable generative repair manifests found")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "repair_fidelity_summary.json").write_text(json.dumps(summary, indent=2))
    if sum(len(item) for item in viz_features) >= 4:
        features = np.concatenate(viz_features); groups = np.asarray(viz_groups); kinds = np.asarray(viz_kinds)
        if len(features) > 2000:
            features, groups, kinds = features[:2000], groups[:2000], kinds[:2000]
        z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(2, (len(features)-1)//3)), random_state=0).fit_transform(features)
        plt.figure(figsize=(7, 5))
        for kind, marker in (("anchor", "x"), ("generated", "o")):
            mask = kinds == kind
            plt.scatter(z[mask, 0], z[mask, 1], s=8, marker=marker, alpha=.65, label=kind)
        plt.legend(); plt.tight_layout(); plt.savefig(output / "fixed_dinov3_anchor_repair_tsne.png", dpi=220)


if __name__ == "__main__":
    main()
