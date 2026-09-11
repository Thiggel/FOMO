#!/usr/bin/env python3
"""The geometry figure, drawn so that density survives the projection.

t-SNE matches a perplexity per point, which normalizes local density away by
construction, so a t-SNE of this embedding shows an evenly filled cloud whatever
the underlying density is.  Reading sparse regions off one is a category error.
densMAP carries a density term in its objective and keeps the variation visible,
and the score distribution is plotted directly beside it so the claim does not
rest on anyone's perception of a scatter plot.

Panels:
  A  densMAP projection of the fixed original panel, coloured by kNN radius,
     with the acquisition band and the selected anchors marked.
  B  the kNN radius distribution itself, repaired against unrepaired, on the
     same panel of original images at the same optimizer stage.
  C  radius of the treated anchors against matched untreated controls, over
     cycles, within the repaired run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY = "#b8b8b8"
BAND = "#4c72b0"
ANCHOR = "#c44e52"
CTRL = "#8c8c8c"


def cycle_files(run: str) -> list[tuple[int, str]]:
    fs = glob.glob(os.path.join(run, "**", "cycle_*_samples.npz"), recursive=True)
    out = [(int(os.path.basename(f).split("cycle_")[1].split("_")[0]), f) for f in fs]
    return sorted(out)


def anchors(run: str, cycle: int) -> set[int]:
    path = os.path.join(run, "generated", "repair_manifests", f"cycle_{cycle}.json")
    if not os.path.exists(path):
        return set()
    rows = json.load(open(path))["rows"]
    return {int(r["anchor_underlying_index"]) for r in rows if r.get("anchor_is_original")}


def matched_controls(underlying, radii, treated: set[int]) -> set[int]:
    """Nearest untreated image by initial radius, without replacement."""
    tmask = np.isin(underlying, list(treated))
    pool = np.where(~tmask)[0]
    order = pool[np.argsort(radii[pool])]
    sorted_values = radii[order]
    used: set[int] = set()
    picked: list[int] = []
    for value in radii[tmask]:
        start = int(np.searchsorted(sorted_values, value))
        for offset in sorted(range(len(order)), key=lambda i: abs(i - start)):
            if offset not in used:
                used.add(offset)
                picked.append(int(underlying[order[offset]]))
                break
    return set(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", required=True, help="run directory of the repaired arm")
    ap.add_argument("--control", required=True, help="run directory of the no-repair arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep = cycle_files(args.repaired)
    ctl = cycle_files(args.control)
    if not rep or not ctl:
        raise SystemExit("missing per-cycle diagnostics in one of the two runs")

    first = np.load(rep[0][1])
    last_rep = np.load(rep[-1][1])
    last_ctl = np.load(ctl[-1][1])
    treated = anchors(args.repaired, rep[0][0])

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))

    # --- A: densMAP, coloured by radius
    import umap

    features = first["normalized_features"].astype(np.float32)
    radii = first["normalized_radii"].astype(float)
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(features), size=min(args.budget, len(features)), replace=False)
    embedding = umap.UMAP(
        densmap=True, n_neighbors=30, min_dist=0.1, random_state=args.seed
    ).fit_transform(features[take])
    r = radii[take]
    lo, hi = np.quantile(radii, [0.75, 0.99])
    in_band = (r >= lo) & (r <= hi)
    is_anchor = np.isin(first["underlying_indices"][take], list(treated))
    ax = axes[0]
    ax.scatter(*embedding[~in_band].T, s=2, c=GREY, linewidths=0, label="below q75")
    ax.scatter(*embedding[in_band & ~is_anchor].T, s=2, c=BAND, linewidths=0,
               label="q75--q99 band")
    ax.scatter(*embedding[is_anchor].T, s=9, c=ANCHOR, linewidths=0, label="selected")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("densMAP of the source panel", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", frameon=False, markerscale=2.5)

    # --- B: radius distribution, repaired against unrepaired
    ax = axes[1]
    for z, colour, label in ((last_ctl, CTRL, "no repair"), (last_rep, BAND, "with repair")):
        v = z["normalized_radii"].astype(float)
        v = v / np.median(v)
        ax.hist(v, bins=80, range=(0, 3), histtype="step", density=True,
                color=colour, label=label, lw=1.3)
    ax.set_xlabel("$k$NN radius / median", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title("Local support, same panel", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    # --- C: treated against matched controls over cycles
    ax = axes[2]
    u0 = first["underlying_indices"]
    r0 = first["normalized_radii"].astype(float)
    controls = matched_controls(u0, r0, treated)
    xs, ts, cs = [], [], []
    for cycle, path in rep:
        z = np.load(path)
        u = z["underlying_indices"]
        v = z["normalized_radii"].astype(float)
        med = np.median(v)
        xs.append(cycle)
        ts.append(v[np.isin(u, list(treated))].mean() / med)
        cs.append(v[np.isin(u, list(controls))].mean() / med)
    ax.plot(xs, ts, marker="o", ms=4, color=ANCHOR, label="treated anchors")
    ax.plot(xs, cs, marker="s", ms=4, color=CTRL, label="matched controls")
    ax.set_xlabel("repair cycle", fontsize=8)
    ax.set_ylabel("radius / median", fontsize=8)
    ax.set_title("Targeted regions over cycles", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_xticks(xs)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"panel B: control median-normalised p90 "
          f"{np.quantile(last_ctl['normalized_radii'] / np.median(last_ctl['normalized_radii']), 0.9):.3f}, "
          f"repaired {np.quantile(last_rep['normalized_radii'] / np.median(last_rep['normalized_radii']), 0.9):.3f}")


if __name__ == "__main__":
    main()
