#!/usr/bin/env python3
"""Report every downstream metric separately as mean plus/minus std."""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    ("Cars", "cars_test_accuracy", "carsknn_knn_test_accuracy"),
    ("Aircraft", "aircraft_test_accuracy", "aircraftknn_knn_test_accuracy"),
    ("Flowers", "flowers_test_accuracy", "flowersknn_knn_test_accuracy"),
    ("Pets", "pets_test_accuracy", "petsknn_knn_test_accuracy"),
    ("CIFAR-10", "cifar10r_test_accuracy", "cifar10knn_knn_test_accuracy"),
    ("CIFAR-100", "cifar100r_test_accuracy", "cifar100knn_knn_test_accuracy"),
    (
        "ImageNet-100-LT",
        "imagenet100lt_test_accuracy",
        "imagenet100ltknn_knn_test_accuracy",
    ),
]


def load_results(experiment_dir: Path, expected_seeds: int) -> list[dict]:
    result_paths = sorted(experiment_dir.glob("seed_*/result.json"))
    if len(result_paths) != expected_seeds:
        raise RuntimeError(
            f"Expected {expected_seeds} seed results in {experiment_dir}, "
            f"found {len(result_paths)}"
        )
    return [json.loads(path.read_text()) for path in result_paths]


def summarize(values: list[float]) -> str:
    percentages = np.asarray(values, dtype=float) * 100.0
    return f"{percentages.mean():.2f} ± {percentages.std():.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing seed_0/result.json, seed_1/result.json, ...",
    )
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    results = load_results(args.experiment_dir, args.expected_seeds)
    print("| Dataset | Linear probe | kNN |")
    print("|---|---:|---:|")
    for dataset, linear_metric, knn_metric in METRICS:
        missing = [
            metric
            for metric in (linear_metric, knn_metric)
            if any(metric not in result for result in results)
        ]
        if missing:
            raise RuntimeError(
                f"Incomplete full metric suite for {dataset}: "
                + ", ".join(missing)
            )
        linear = summarize([result[linear_metric] for result in results])
        knn = summarize([result[knn_metric] for result in results])
        print(f"| {dataset} | {linear} | {knn} |")


if __name__ == "__main__":
    main()
