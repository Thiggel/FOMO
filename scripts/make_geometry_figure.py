#!/usr/bin/env python3
"""Where the acquisition budget lands, and what it does to local support.

Panel A projects the cycle-0 source panel with densMAP, coloured by kNN
radius, with the acquisition band and the selected anchors overlaid. The
projection is an illustration. A two-dimensional embedding cannot preserve
2048-dimensional neighbourhood radii, so the quantitative statement about
where anchors sit is panel B, the percentile of every anchor's radius within
the panel. Panel C is the fixed-space intervention test from the appendix
table, drawn so the paired treated-against-control comparison is visible.

The cycle-0 arrays come from the shared branch checkpoint every arm starts
from, so they describe the representation before any repair.
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, BAND, ANCHOR, CTRL = "#b8b8b8", "#3b6fb6", "#c9413a", "#6e6e6e"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with cycle-0 diagnostics and manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    z = np.load(os.path.join(a.run, "generated/representation_diagnostics/cycle_0_samples.npz"))
    feats = z["normalized_features"].astype(np.float32)
    radii = z["normalized_radii"].astype(float)
    under = z["underlying_indices"]
    rows = json.load(open(os.path.join(a.run, "generated/repair_manifests/cycle_0.json")))["rows"]
    anchors = {r["anchor_underlying_index"] for r in rows}
    is_anchor = np.isin(under, list(anchors))
    lo, hi = np.quantile(radii, [0.75, 0.99])
    in_band = (radii >= lo) & (radii <= hi)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # A: densMAP coloured by radius
    import umap
    rng = np.random.default_rng(a.seed)
    take = rng.choice(len(feats), size=min(a.budget, len(feats)), replace=False)
    take = np.union1d(take, np.where(is_anchor)[0])
    emb = umap.UMAP(densmap=True, n_neighbors=30, min_dist=0.1, random_state=a.seed).fit_transform(feats[take])
    r = radii[take]; ia = is_anchor[take]; ib = in_band[take]
    ax = axes[0]
    order = np.argsort(r)
    vlo, vhi = np.quantile(radii, [0.02, 0.995])
    sc = ax.scatter(*emb[order].T, c=r[order], s=4, alpha=0.55, cmap="viridis", vmin=vlo, vmax=vhi, linewidths=0, rasterized=True)
    # Anchors carry their own radius colour with a dark edge, so the reader can
    # see that they are high-radius points even where the projection places
    # them inside the visually dense core.
    ax.scatter(*emb[ia].T, c=r[ia], cmap="viridis", vmin=vlo, vmax=vhi, s=22,
               edgecolors=ANCHOR, linewidths=0.6, label="selected anchors")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("$k$NN radius", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("densMAP of the source panel at cycle 0", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", frameon=False)

    # B: anchor radius percentile within the panel
    ax = axes[1]
    pct = (radii[:, None] < radii[is_anchor][None, :]).mean(0)
    ax.hist(pct, bins=np.linspace(0, 1, 41), color=ANCHOR, alpha=0.85)
    ax.axvspan(0.75, 0.99, color=BAND, alpha=0.15, lw=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("percentile of anchor radius in the panel", fontsize=8)
    ax.set_ylabel("anchors", fontsize=8)
    ax.set_title(f"{int(is_anchor.sum())} anchors, {100*((pct>=0.75)&(pct<=0.99)).mean():.0f}% inside q75 to q99", fontsize=9)
    ax.tick_params(labelsize=7)

    # C: fixed-space treated against matched controls, mean +- sd over three seeds
    ax = axes[2]
    labels = ["treated\nanchors", "matched\ncontrols"]
    means = np.array([-0.00273, -0.00139]); sds = np.array([0.00062, 0.00004])
    ax.bar(labels, means * 1e3, yerr=sds * 1e3, color=[ANCHOR, CTRL], capsize=4, width=0.55)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(r"change in $k$NN radius ($\times 10^{-3}$)", fontsize=8)
    ax.set_title("Fixed encoder, after inserting all repairs", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.text(0.98, 0.04, "radius difference $-1.34\\pm0.67$\npurity difference $+0.0134\\pm0.0032$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", dpi=200)
    print("wrote", a.out, "| anchors in band:", f"{((pct>=0.75)&(pct<=0.99)).mean():.3f}")

if __name__ == "__main__":
    main()
