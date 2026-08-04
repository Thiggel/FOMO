"""Aggregate selector percentiles, overlap, and synthetic-anchor provenance."""
import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def metadata(path: Path):
    # .../<family>/<condition>/<seed>/generated/ood_diagnostics/<cycle>/selection.npz
    generated = path.parents[2]
    return {
        "family": generated.parents[2].name,
        "condition": generated.parents[1].name,
        "seed": generated.parent.name,
        "cycle": path.parent.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="recursive glob for selection.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    selections = {}
    for raw_path in sorted(glob.glob(args.inputs, recursive=True)):
        path = Path(raw_path)
        values = np.load(path)
        if "selected_score_percentiles" not in values:
            continue
        info = metadata(path)
        percentiles = values["selected_score_percentiles"]
        original = values["selected_is_original"].astype(float)
        key = (info["family"], info["condition"], info["seed"], info["cycle"])
        selections[key] = set(values["selected_underlying_indices"].astype(int).tolist())
        records.append(
            {
                **info,
                "n_selected": int(len(percentiles)),
                "selected_percentile_mean": float(percentiles.mean()),
                "selected_percentile_median": float(np.median(percentiles)),
                "selected_percentile_p10": float(np.quantile(percentiles, 0.10)),
                "selected_percentile_p90": float(np.quantile(percentiles, 0.90)),
                "selected_original_fraction": float(original.mean()),
            }
        )
    if not records:
        raise RuntimeError("No selection manifests found")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with open(output / "selection_provenance_summary.json", "w") as handle:
        json.dump(records, handle, indent=2)

    labels = [f"{item['family']}/{item['condition']}" for item in records]
    plt.figure(figsize=(max(7, len(records) * .35), 4))
    plt.scatter(range(len(records)), [item["selected_percentile_median"] for item in records], s=14)
    plt.xticks(range(len(records)), labels, rotation=85, fontsize=6)
    plt.ylabel("median selected score percentile")
    plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(output / "selected_score_percentiles.png", dpi=220)
    plt.close()

    plt.figure(figsize=(max(7, len(records) * .35), 4))
    plt.bar(range(len(records)), [item["selected_original_fraction"] for item in records])
    plt.xticks(range(len(records)), labels, rotation=85, fontsize=6)
    plt.ylabel("original-anchor fraction"); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(output / "anchor_provenance.png", dpi=220)
    plt.close()

    overlap = []
    grouped = {}
    for key, selected in selections.items():
        family, _, seed, cycle = key
        grouped.setdefault((family, seed, cycle), []).append((key[1], selected))
    for group_key, variants in grouped.items():
        for (left_name, left), (right_name, right) in combinations(variants, 2):
            union = left | right
            overlap.append({
                "family": group_key[0], "seed": group_key[1], "cycle": group_key[2],
                "left": left_name, "right": right_name,
                "jaccard": float(len(left & right) / len(union)) if union else 1.0,
            })
    with open(output / "selection_jaccard.json", "w") as handle:
        json.dump(overlap, handle, indent=2)


if __name__ == "__main__":
    main()
