"""Controlled 2D toy test of recoverable sparse-support allocation.

Rare coherent points can be repaired by a local generator; isolated noise
points cannot.  The experiment tests the falsifiable interior-window premise
without claiming to model the full neural SSL pipeline.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def choose(points, score, budget, policy, rng):
    if policy == "uniform":
        return rng.choice(len(points), budget, replace=False)
    if policy == "top_tail":
        return np.argsort(score)[-budget:]
    order = np.argsort(score)
    lower, upper = np.quantile(score, [.75, .99])
    trimmed = order[(score[order] >= lower) & (score[order] <= upper)]
    pool = min(len(trimmed), budget * 4)
    widths = score[trimmed[pool - 1:]] - score[trimmed[:len(trimmed) - pool + 1]]
    start = int(np.argmin(widths)) if len(widths) else 0
    candidates = trimmed[start:start + pool]
    # Greedy farthest-first coverage within the densest sparse score window.
    picked = [int(candidates[0])]
    while len(picked) < min(budget, len(candidates)):
        distance = ((points[candidates, None] - points[np.asarray(picked)][None]) ** 2).sum(-1).min(1)
        picked.append(int(candidates[np.argmax(distance)]))
    return np.asarray(picked)


def trial(noise_count, seed):
    rng = np.random.default_rng(seed)
    head = rng.normal([0, 0], .55, (1000, 2)); rare = rng.normal([3, 0], .30, (60, 2))
    noise = rng.uniform([-6, -6], [6, 6], (noise_count, 2))
    x = np.vstack([head, rare, noise]); y = np.r_[np.zeros(len(head)), np.ones(len(rare)), -np.ones(len(noise))]
    knn = NearestNeighbors(n_neighbors=11).fit(x); distance, _ = knn.kneighbors(x); score = distance[:, 1:].mean(1)
    test_head = rng.normal([0, 0], .55, (500, 2)); test_rare = rng.normal([3, 0], .30, (500, 2))
    test_x = np.vstack([test_head, test_rare]); test_y = np.r_[np.zeros(500), np.ones(500)]
    rows = []
    for policy in ("uniform", "top_tail", "densest_window"):
        selected = choose(x, score, 80, policy, rng)
        # Only coherent semantic anchors are repairable; generated variants of
        # contaminants carry no useful class support.
        coherent = selected[y[selected] >= 0]
        generated = x[coherent] + rng.normal(0, .18, (len(coherent), 2))
        generated_y = y[coherent]
        train_x = np.vstack([x[y >= 0], generated]); train_y = np.r_[y[y >= 0], generated_y]
        classifier = KNeighborsClassifier(n_neighbors=9).fit(train_x, train_y)
        pred = classifier.predict(test_x)
        rows.append({"noise_count": noise_count, "seed": seed, "policy": policy,
                     "rare_accuracy": float((pred[500:] == test_y[500:]).mean()),
                     "coverage_error": float(np.linalg.norm(generated.mean(0) - np.array([3, 0])) if len(generated) else 10),
                     "wasted_noise_fraction": float((y[selected] < 0).mean())})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--seeds", type=int, default=30); args = parser.parse_args()
    rows = [row for noise in (0, 20, 50, 100, 200) for seed in range(args.seeds) for row in trial(noise, seed)]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); (output / "toy_recoverable_support.json").write_text(json.dumps(rows, indent=2))
    plt.figure(figsize=(6, 4))
    for policy in ("uniform", "top_tail", "densest_window"):
        values = [np.mean([r["rare_accuracy"] for r in rows if r["noise_count"] == noise and r["policy"] == policy]) for noise in (0,20,50,100,200)]
        plt.plot((0,20,50,100,200), values, marker="o", label=policy)
    plt.xlabel("isolated-noise points"); plt.ylabel("rare-mode kNN accuracy"); plt.legend(); plt.tight_layout(); plt.savefig(output / "toy_recoverable_support.png", dpi=220)


if __name__ == "__main__":
    main()
