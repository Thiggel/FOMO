#!/usr/bin/env python3
"""Where the acquisition budget lands in the score distribution.

One ImageNet-100-LT run at cycle 0.  The full score histogram is drawn in
grey, the 75th and 99th percentiles are marked, the candidate pool (the 4B
scores closest to the modal bin of the band-restricted histogram) is shaded,
the top-tail control (the B largest scores) is shaded in red, and the scores of
the B anchors actually selected are overlaid as a histogram.  The pool logic
mirrors experiment/ood/ood.py, so the figure shows what the selector does.
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, POOL, ANCHOR, TAIL = "#c8c8c8", "#2a9d8f", "#c9413a", "#e76f51"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--pool-multiplier", type=int, default=4)
    ap.add_argument("--band", type=float, nargs=2, default=(0.75, 0.99))
    a = ap.parse_args()
    z = np.load(os.path.join(a.run, "generated/representation_diagnostics/cycle_0_samples.npz"))
    d = z["normalized_radii"].astype(float); under = z["underlying_indices"]
    rows = json.load(open(os.path.join(a.run, "generated/repair_manifests/cycle_0.json")))["rows"]
    anchors = np.isin(under, list({r["anchor_underlying_index"] for r in rows}))
    lo, hi = np.quantile(d, a.band)
    band = d[(d >= lo) & (d <= hi)]
    edges = np.histogram_bin_edges(band, bins="auto")
    nb = max(8, min(512, len(edges) - 1))
    counts, edges = np.histogram(band, bins=nb, range=(lo, hi))
    m = int(np.argmax(counts)); centre = 0.5 * (edges[m] + edges[m + 1])
    in_band = np.where((d >= lo) & (d <= hi))[0]
    pool = in_band[np.argsort(np.abs(d[in_band] - centre))[: a.pool_multiplier * a.budget]]
    tail_lo = np.sort(d)[-a.budget]
    xmax = np.quantile(d, 0.998)

    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    bins = np.linspace(d.min(), xmax, 70)
    ax.hist(d[d <= xmax], bins=bins, color=GREY, edgecolor="none", label="all source images")
    ax.axvspan(d[pool].min(), d[pool].max(), color=POOL, alpha=0.25, lw=0,
               label=f"candidate pool ({a.pool_multiplier}B nearest the band mode)")
    ax.axvspan(tail_lo, xmax, color=TAIL, alpha=0.22, lw=0, label="top-tail control (B largest)")
    ax.hist(d[anchors], bins=bins, color=ANCHOR, edgecolor="none", alpha=0.9, label="selected anchors (B=500)")
    for q, lab in ((lo, "q75"), (hi, "q99")):
        ax.axvline(q, color="k", lw=0.8, ls="--")
        ax.text(q, 1.01, lab, ha="center", va="bottom", fontsize=7, fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("mean $k$NN distance (score)", fontsize=8); ax.set_ylabel("images", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight", dpi=200)
    pct = (d[:, None] < d[anchors][None, :]).mean(0)
    print(f"wrote {a.out}: band [{lo:.4f},{hi:.4f}] mode {centre:.4f} pool [{d[pool].min():.4f},{d[pool].max():.4f}] "
          f"anchors in band {((pct>=a.band[0])&(pct<=a.band[1])).mean():.3f} median pct {np.median(pct):.2f}")

if __name__ == "__main__":
    main()
