#!/usr/bin/env python3
"""Score distribution over the repair cycles of one run, on one shared axis.

The claim is that the distribution keeps its right-skewed shape while its scale
grows, so the acquisition band must be re-derived from the current
representation.  A shared x-axis makes the drift visible without reading tick
labels.  The marked line is the modal bin of the histogram restricted to the
75th to 99th percentile band, which is what the selector centers on.

With --baseline the same panels are drawn for a run of the same source and seed
that is trained for the same number of cycles without any repair, so the drift
of the band can be read against a no-repair control.
"""
from __future__ import annotations
import argparse, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, ANCHOR = "#c8c8c8", "#c9413a"


def load(run: str, cycles: int) -> list[np.ndarray]:
    files = sorted(glob.glob(os.path.join(run, "generated/representation_diagnostics/cycle_*_samples.npz")),
                   key=lambda p: int(os.path.basename(p).split("_")[1]))[: cycles]
    return [np.load(f)["normalized_radii"].astype(float) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None, help="run root of a no-repair run of the same source and seed")
    ap.add_argument("--cycles", type=int, default=4)
    a = ap.parse_args()
    rows = [("with repair", load(a.run, a.cycles))]
    if a.baseline:
        rows.append(("without repair", load(a.baseline, a.cycles)))
    flat = [d for _, ds in rows for d in ds]
    xmax = max(np.quantile(d, 0.998) for d in flat)
    bins = np.linspace(min(d.min() for d in flat), xmax, 80)
    ncol = max(len(ds) for _, ds in rows)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.65 * ncol, 1.35 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for r, (label, data) in enumerate(rows):
        for i, d in enumerate(data):
            ax = axes[r][i]
            lo, hi = np.quantile(d, [0.75, 0.99])
            ax.hist(d[d <= xmax], bins=bins, color=GREY, edgecolor="none")
            ax.axvspan(lo, hi, color=ANCHOR, alpha=0.18, lw=0)
            band = d[(d >= lo) & (d <= hi)]
            edges = np.histogram_bin_edges(band, bins="auto"); nb = max(8, min(512, len(edges) - 1))
            counts, edges = np.histogram(band, bins=nb, range=(lo, hi)); m = int(np.argmax(counts))
            ax.axvline(0.5 * (edges[m] + edges[m + 1]), color=ANCHOR, lw=1.3)
            if r == 0:
                ax.set_title(f"cycle {i}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if r == nrow - 1:
                ax.set_xlabel("mean $k$NN distance", fontsize=8)
        axes[r][0].set_ylabel((label + "\nimages") if nrow > 1 else "images", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight", dpi=200)
    for label, data in rows:
        print(label, "q75:", [round(float(np.quantile(d, 0.75)), 3) for d in data],
              "q99:", [round(float(np.quantile(d, 0.99)), 3) for d in data])


if __name__ == "__main__":
    main()
