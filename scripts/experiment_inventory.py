#!/usr/bin/env python3
"""Report what still has to run, derived from result files rather than job logs.

Job liveness says nothing about whether a result was written: the expensive
failure mode in this project is a run that trains for hours and then dies in
the downstream benchmark stage.  This script therefore treats
``result.json`` as the only source of truth and classifies every seed
directory as complete, backfillable (a finished checkpoint whose result is
missing paper metrics) or unrecoverable (no checkpoint to evaluate).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Spelled out rather than imported from FinetuningBenchmarks, which pulls in
# torch and lightning: this tooling has to run on login nodes that have no GPU
# stack installed.  test_inventory_metrics_match_the_paper_suite keeps this
# list identical to the benchmark contract.
PAPER_METRICS = [
    "cars_test_accuracy",
    "carsknn_knn_test_accuracy",
    "aircraft_test_accuracy",
    "aircraftknn_knn_test_accuracy",
    "flowers_test_accuracy",
    "flowersknn_knn_test_accuracy",
    "pets_test_accuracy",
    "petsknn_knn_test_accuracy",
    "cifar10r_test_accuracy",
    "cifar10knn_knn_test_accuracy",
    "cifar100r_test_accuracy",
    "cifar100knn_knn_test_accuracy",
    "imagenet100lt_test_accuracy",
    "imagenet100ltknn_knn_test_accuracy",
]


def missing_metrics(result_path: Path) -> list[str] | None:
    """Return the paper metrics absent from a result, or None if unreadable."""
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return [
        metric
        for metric in PAPER_METRICS
        if not isinstance(payload.get(metric), (int, float))
    ]


def scan(checkpoint_root: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)

    for result_path in checkpoint_root.glob("*/*/seed_*/result.json"):
        seed_dir = result_path.parent
        experiment = result_path.relative_to(checkpoint_root).parts[0]
        dataset_dir = result_path.relative_to(checkpoint_root).parts[1]
        seed = seed_dir.name.removeprefix("seed_")
        absent = missing_metrics(result_path)

        cell = {
            "experiment": experiment,
            "dataset_dir": dataset_dir,
            "seed": seed,
            "missing": absent,
            "seed_dir": str(seed_dir),
        }

        if absent is None:
            buckets["unreadable"].append(cell)
        elif not absent:
            buckets["complete"].append(cell)
        elif (seed_dir / "last.ckpt").is_file():
            buckets["backfillable"].append(cell)
        else:
            buckets["no_checkpoint"].append(cell)

    return buckets


def write_manifest(cells: list[dict], path: Path) -> None:
    """Write the tab-separated manifest consumed by the backfill queue."""
    lines = [
        f"{cell['experiment']}\t{cell['seed']}\t{cell['dataset_dir']}"
        for cell in sorted(cells, key=lambda c: (c["experiment"], c["seed"]))
    ]
    path.write_text("\n".join(lines) + "\n" if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("CHECKPOINT_ROOT_DIR", "checkpoints")),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write the backfillable cells to this manifest path",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="List every cell, not just totals"
    )
    args = parser.parse_args()

    buckets = scan(args.checkpoint_root)

    complete_by_experiment: dict[str, int] = defaultdict(int)
    for cell in buckets["complete"]:
        complete_by_experiment[cell["experiment"]] += 1
    finished_experiments = sum(
        1 for count in complete_by_experiment.values() if count >= 3
    )

    print(f"checkpoint root: {args.checkpoint_root}")
    print(f"complete seed results:      {len(buckets['complete'])}")
    print(f"  experiments with 3 seeds: {finished_experiments}")
    print(f"backfillable (has last.ckpt): {len(buckets['backfillable'])}")
    print(f"missing checkpoint:           {len(buckets['no_checkpoint'])}")
    print(f"unreadable results:           {len(buckets['unreadable'])}")

    if args.verbose:
        for cell in sorted(
            buckets["backfillable"], key=lambda c: (c["experiment"], c["seed"])
        ):
            print(
                f"  BACKFILL {cell['experiment']} seed {cell['seed']} "
                f"({len(cell['missing'])} metrics missing)"
            )
        for cell in sorted(
            buckets["no_checkpoint"], key=lambda c: (c["experiment"], c["seed"])
        ):
            print(f"  NO CHECKPOINT {cell['experiment']} seed {cell['seed']}")

    if args.manifest:
        write_manifest(buckets["backfillable"], args.manifest)
        print(f"\nwrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
