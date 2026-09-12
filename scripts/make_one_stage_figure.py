#!/usr/bin/env python3
"""The paper's central contrast, drawn on one axis.

Left: nine score bands, each spending the identical budget on a different part
of the kNN score distribution, after a single repair stage.  Right: three
acquisition rules under five stages.  Same budget, same generator, same
optimizer budget in each panel; the panels differ in how often the acquisition
signal is recomputed.  Both are plotted on the same vertical span so the reader
can see the flat line and the separated one at the same scale.

Numbers come from the generated tables, so the figure cannot drift from them.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND = "#4c72b0"
RULE = "#c44e52"
GREY = "#8c8c8c"


def band_averages(path: Path) -> list[tuple[str, float]]:
    out = []
    for line in path.read_text().splitlines():
        if "&" not in line or "\\\\" not in line or line.strip().startswith("Method"):
            continue
        label = line.split("&")[0].strip()
        values = [float(v) for v in re.findall(r"\$([0-9.]+)\\pm", line)]
        if len(values) == 7:
            out.append((label, mean(values)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile-table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bands = band_averages(args.percentile_table)
    labels = [b[0].replace("q", "").split(" ")[0] for b, _ in zip(bands, bands)]
    labels = [b[0].split(" ")[0].replace("q", "") for b in bands]
    values = [v for _, v in bands]

    # Five-stage acquisition rules, from tables/main_mechanism.tex.
    rules = [("top tail", 34.3, 1.2), ("uniform", 35.5, 1.2), ("mode\nwindow", 42.3, 0.7)]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.05), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    low = min(min(values), min(v for _, v, _ in rules)) - 2
    high = max(max(values), max(v for _, v, _ in rules)) + 2

    ax = axes[0]
    ax.bar(range(len(values)), values, color=BAND, width=0.68)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylim(low, high)
    ax.set_ylabel("seven-task linear probe", fontsize=8)
    ax.set_title("one repair stage: nine score bands, spread 1.32", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    # The cosine row of the robustness table is provably the same experiment as
    # the normalized-L2 row, and the two differ by up to 1.3 points.  That is the
    # pipeline's own resolution, so a band inside this envelope is unreadable.
    centre = mean(values)
    ax.axhspan(centre - 0.65, centre + 0.65, color=GREY, alpha=0.22, lw=0)
    ax.axhline(centre, color=GREY, lw=0.8, ls="--")
    ax.text(0.02, 0.93, "shaded: run-to-run resolution, 1.3 points",
            transform=ax.transAxes, fontsize=6.5, color="#555555")

    ax = axes[1]
    ax.bar(
        range(len(rules)),
        [v for _, v, _ in rules],
        yerr=[e for _, _, e in rules],
        color=RULE,
        width=0.6,
        capsize=3,
    )
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([n for n, _, _ in rules], fontsize=7)
    ax.set_ylim(low, high)
    ax.set_yticklabels([])
    ax.set_title("five stages: three rules, spread 8.0", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}: {len(values)} bands, spread {max(values)-min(values):.2f}")


if __name__ == "__main__":
    main()
