#!/usr/bin/env python3
"""Figures showing how the representation changes across repair cycles.

Reviewers of the previous submission asked for evidence of the geometric
claim beyond a single histogram: per-cycle score distributions and a 2D
projection of the embedding space showing that the targeted region actually
densifies.  This script draws both from the per-cycle diagnostics the training
loop already writes.

Those diagnostics are recorded from the live encoder at the end of each cycle,
so they are unaffected by the checkpoint-callback fault that invalidated some
published tables: that fault changed which weights were *saved*, not which
were used to score and select during the run.

Inputs are the ``cycle_*_samples.npz`` files under
``generated/representation_diagnostics`` and the anchor lists under
``generated/repair_manifests``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

BAND_LO, BAND_HI = 0.75, 0.99
GREY, ANCHOR, GENERATED = "#c8c8c8", "#d1495b", "#2e7dd1"


def load_cycles(run: Path):
    """Return per-cycle (features, radii, underlying indices, anchor set)."""
    diag = run / "generated" / "representation_diagnostics"
    man = run / "generated" / "repair_manifests"
    cycles = []
    for path in sorted(diag.glob("cycle_*_samples.npz"),
                       key=lambda p: int(p.stem.split("_")[1])):
        idx = int(path.stem.split("_")[1])
        blob = np.load(path, allow_pickle=True)
        anchors: set[int] = set()
        manifest = man / f"cycle_{idx}.json"
        if manifest.exists():
            rows = json.loads(manifest.read_text())["rows"]
            anchors = {int(r["anchor_underlying_index"]) for r in rows}
        cycles.append({
            "cycle": idx,
            "features": blob["normalized_features"].astype(np.float32),
            "radii": blob["normalized_radii"].astype(np.float32),
            "underlying": blob["underlying_indices"].astype(np.int64),
            "anchors": anchors,
        })
    return cycles


def figure_histograms(cycles, out: Path) -> None:
    """Score distribution per cycle with the acquisition band marked."""
    n = len(cycles)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cycles):
        d = c["radii"]
        lo, hi = np.quantile(d, BAND_LO), np.quantile(d, BAND_HI)
        counts, edges, _ = ax.hist(d, bins=60, color=GREY, edgecolor="none")
        ax.axvspan(lo, hi, color=ANCHOR, alpha=0.16, lw=0)
        inside = (edges[:-1] >= lo) & (edges[:-1] <= hi)
        if inside.any():
            mode = edges[:-1][inside][int(np.argmax(counts[inside]))]
            ax.axvline(mode, color=ANCHOR, lw=1.4)
        ax.set_title(f"cycle {c['cycle']}", fontsize=9)
        ax.set_xlabel("mean $k$NN distance", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("images", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_tsne(cycles, out: Path, budget: int, seed: int) -> None:
    """2D projection per cycle, marking anchors and added samples.

    The projection is fitted separately per cycle because the encoder changes
    between cycles, so the coordinates are not comparable across panels.  What
    is comparable is where the selected anchors sit within each panel and how
    the added samples are placed relative to them.
    """
    rng = np.random.default_rng(seed)
    original = set(cycles[0]["underlying"].tolist())
    n = len(cycles)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cycles):
        keep = np.arange(len(c["underlying"]))
        if len(keep) > budget:
            keep = rng.choice(keep, budget, replace=False)
        feats, under = c["features"][keep], c["underlying"][keep]
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=seed, max_iter=750).fit_transform(feats)
        # Colour by the acquisition score, not by anchor membership.  FPS
        # deliberately spreads the anchors across the candidate pool for
        # diversity, so plotting the anchors alone shows a near-uniform
        # scatter and hides the band they were drawn from.
        radii = c["radii"][keep]
        lo, hi = np.quantile(radii, BAND_LO), np.quantile(radii, BAND_HI)
        band = (radii >= lo) & (radii <= hi)
        ax.scatter(emb[~band, 0], emb[~band, 1], s=2, c=GREY, lw=0,
                   label="below q75")
        ax.scatter(emb[band, 0], emb[band, 1], s=4, c=GENERATED, lw=0,
                   alpha=0.75, label="acquisition band q75--q99")
        is_anchor = np.array([u in c["anchors"] for u in under])
        if is_anchor.any():
            ax.scatter(emb[is_anchor, 0], emb[is_anchor, 1], s=11, c=ANCHOR,
                       lw=0, label="selected anchor (after FPS)")
        ax.set_title(f"cycle {c['cycle']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=6, frameon=False, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_treated_control(cycles, out: Path) -> None:
    """Score of treated anchors against matched untreated images over cycles.

    Controls are matched per cycle to the anchors of cycle 0 by initial score,
    so both groups start from the same place and any later separation is
    attributable to the repairs.
    """
    first = cycles[0]
    if not first["anchors"]:
        print("no anchors in cycle 0; skipping treated/control figure")
        return
    treated_ids = first["anchors"]
    pos = {u: i for i, u in enumerate(first["underlying"].tolist())}
    t_idx = [pos[u] for u in treated_ids if u in pos]
    t_scores = first["radii"][t_idx]

    candidates = [i for u, i in pos.items() if u not in treated_ids]
    cand_scores = first["radii"][candidates]
    order = np.argsort(cand_scores)
    sorted_c = cand_scores[order]
    used, control_ids = set(), []
    for s in t_scores:  # nearest unused control by initial score
        j = int(np.searchsorted(sorted_c, s))
        for cand in sorted(range(len(order)), key=lambda x: abs(x - j)):
            if cand not in used:
                used.add(cand)
                control_ids.append(first["underlying"][candidates[order[cand]]])
                break
    control_ids = set(int(x) for x in control_ids)

    xs, t_mean, c_mean, t_err, c_err = [], [], [], [], []
    for c in cycles:
        p = {u: i for i, u in enumerate(c["underlying"].tolist())}
        ti = [p[u] for u in treated_ids if u in p]
        ci = [p[u] for u in control_ids if u in p]
        if not ti or not ci:
            continue
        xs.append(c["cycle"])
        t_mean.append(float(np.mean(c["radii"][ti])))
        c_mean.append(float(np.mean(c["radii"][ci])))
        t_err.append(float(np.std(c["radii"][ti]) / np.sqrt(len(ti))))
        c_err.append(float(np.std(c["radii"][ci]) / np.sqrt(len(ci))))

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.errorbar(xs, t_mean, yerr=t_err, marker="o", ms=4, color=ANCHOR,
                capsize=2, label="treated anchors (cycle 0)")
    ax.errorbar(xs, c_mean, yerr=c_err, marker="s", ms=4, color="#4a4a4a",
                capsize=2, label="matched untreated controls")
    ax.set_xlabel("cycle", fontsize=9)
    ax.set_ylabel("mean $k$NN distance", fontsize=9)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for x, t, c in zip(xs, t_mean, c_mean):
        print(f"  cycle {x}: treated {t:.5f}  control {c:.5f}  diff {t - c:+.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tsne-budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cycles = load_cycles(args.run)
    if not cycles:
        raise SystemExit(f"no cycle diagnostics under {args.run}")
    print(f"{len(cycles)} cycles, sizes {[len(c['radii']) for c in cycles]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_histograms(cycles, args.out_dir / "cycle_score_histograms.pdf")
    figure_treated_control(cycles, args.out_dir / "treated_vs_control.pdf")
    figure_tsne(cycles, args.out_dir / "cycle_tsne.pdf",
                args.tsne_budget, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
