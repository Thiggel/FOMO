#!/usr/bin/env python3
"""Does the loop add support where the encoder said it was missing?

Two panels from the frozen cycle-0 encoder, so nothing here reflects a change
of coordinates.  Left: relative change in kNN radius after every generated
image is inserted into the index, for the treated anchors and for controls
matched one-to-one within class on initial radius.  Right: change in
neighborhood label purity for the same two groups.  One marker per seed, bars
are the three-seed mean.  Relative radius is used because the absolute radii
are of order 0.06 and unreadable as raw differences.

Inputs are the per-seed reports written by
paper_work/analysis/fixed_support_injection.py.
"""
from __future__ import annotations
import argparse, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANCHOR, CTRL = "#c9413a", "#8c8c8c"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="glob for seed_*.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reps = [json.load(open(p)) for p in sorted(glob.glob(a.reports))]
    if not reps:
        raise SystemExit("no reports matched")
    rel = lambda g: np.array([100 * r[g]["radius_change"] / r[g]["radius_before"] for r in reps])
    pur = lambda g: np.array([100 * r[g]["purity_change"] for r in reps])
    groups = [("treated anchors", "anchors", ANCHOR), ("matched controls", "matched_controls", CTRL)]

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0), gridspec_kw={"wspace": 0.45})
    for ax, f, ylabel, title in (
        (axes[0], rel, "radius change (%)", "Local radius"),
        (axes[1], pur, "purity change (pts)", "Neighborhood purity"),
    ):
        for i, (label, key, colour) in enumerate(groups):
            v = f(key)
            ax.bar(i, v.mean(), color=colour, width=0.6, alpha=0.9)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v,
                       color="k", s=14, zorder=3, alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["anchors", "controls"], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    d = rel("anchors") - rel("matched_controls")
    axes[0].set_title("local radius", fontsize=8.5)
    p = pur("anchors") - pur("matched_controls")
    axes[1].set_title("neighborhood purity", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", dpi=200)
    print("radius %:", rel("anchors").round(2), rel("matched_controls").round(2), "DiD", d.round(2))
    print("purity pts:", pur("anchors").round(2), pur("matched_controls").round(2), "DiD", p.round(2))

if __name__ == "__main__":
    main()
