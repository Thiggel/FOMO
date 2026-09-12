#!/usr/bin/env python3
"""Per-objective effect of repair, one bar per downstream measurement.

The four-objective table is the paper's evidence that the loop is not tied to
one SSL objective, and read as a table it is 112 numbers.  As differences it is
one picture: 14 measurements per objective, linear probe and kNN on each of the
seven downstream datasets, repaired minus unrepaired.

Numbers are read out of the generated tables so the figure cannot drift.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UP = "#4c72b0"
DOWN = "#c44e52"
OBJECTIVES = ["SimCLR", "MoCo v3", "DINO", "MAE"]
DATASETS = ["Cars", "Aircraft", "Flowers", "Pets", "C10", "C100", "IN100-LT"]


def rows(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        if "&" not in line or "\\\\" not in line or line.strip().startswith("Method"):
            continue
        name = line.split("&")[0].strip()
        values = [float(v) for v in re.findall(r"([0-9.]+)\\pm", line)]
        if len(values) == 7:
            out[name] = values
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", type=Path, required=True)
    ap.add_argument("--knn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lin, knn = rows(args.linear), rows(args.knn)
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.15), sharey=True)
    for ax, objective in zip(axes, OBJECTIVES):
        deltas, labels = [], []
        for table, tag in ((lin, "lp"), (knn, "kNN")):
            base = table[f"{objective}, SSL baseline"]
            repaired = table[f"{objective}, with \\method"]
            deltas += [r - b for b, r in zip(base, repaired)]
            labels += [f"{d} {tag}" for d in DATASETS]
        colours = [UP if d > 0 else DOWN for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=colours, width=0.78)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_xticks([])
        ax.set_title(
            f"{objective}\n{sum(1 for d in deltas if d > 0)}/14 up, "
            f"mean {mean(deltas):+.2f}",
            fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("repaired $-$ unrepaired", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
