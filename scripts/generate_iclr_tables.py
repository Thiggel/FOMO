#!/usr/bin/env python3
"""Regenerate ICLR result tables directly from stored seed results.

The tables in the manuscript were previously transcribed by hand from job
logs, which is why two mutually inconsistent SimCLR rows survived into the
submitted version.  Here each row names the experiments it aggregates, the
numbers come from result.json, and a row whose seeds are not all complete is
rendered as a placeholder instead of silently averaging fewer seeds.

Run it again whenever backfills land; rows fill in as their evidence arrives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.experiment_inventory import PAPER_METRICS

# (column header, linear-probe metric, kNN metric)
DATASETS = [
    ("Cars", "cars_test_accuracy", "carsknn_knn_test_accuracy"),
    ("Aircraft", "aircraft_test_accuracy", "aircraftknn_knn_test_accuracy"),
    ("Flowers", "flowers_test_accuracy", "flowersknn_knn_test_accuracy"),
    ("Pets", "pets_test_accuracy", "petsknn_knn_test_accuracy"),
    ("CIFAR-10", "cifar10r_test_accuracy", "cifar10knn_knn_test_accuracy"),
    ("CIFAR-100", "cifar100r_test_accuracy", "cifar100knn_knn_test_accuracy"),
    (
        "IN100-LT",
        "imagenet100lt_test_accuracy",
        "imagenet100ltknn_knn_test_accuracy",
    ),
]

# Each row lists the experiment directory per seed, in seed order.  Naming is
# irregular across waves, so it is spelled out rather than derived.
POLICY_REPAIR_ROWS = [
    (
        "Adaptive \\method",
        [f"rebuttal_fullpolicy3e60_adaptive_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "Frozen first-cycle selector",
        [f"rebuttal_fullpolicy3e60_static_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "One-shot repair",
        [f"rebuttal_fullpolicy3e60_one_shot_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "Mode-window conventional augmentation",
        [f"rebuttal_fullpolicy3e60_conventional_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "Uniform acquisition",
        [
            "rebuttal_fullpolicy3e60_uniform_0_gruenau_retry1",
            "rebuttal_fullpolicy3e60_uniform_1_gruenau_retry1",
            "rebuttal_fullpolicy3e60_uniform_2_gruenau_retry1",
        ],
    ),
    (
        "Top-tail acquisition",
        [
            "rebuttal_fullpolicy3e60_top_tail_0_iclr2027",
            "rebuttal_fullpolicy3e60_top_tail_1_gruenau_retry1",
            "rebuttal_fullpolicy3e60_top_tail_2_gruenau_retry1",
        ],
    ),
    (
        "AIDE-style VLM acquisition and text-to-image",
        [f"iclr_aide_ssl_clip_cluster_{s}_v1" for s in (0, 1, 2)],
    ),
    (
        "Mode-window caption and text-to-image",
        [f"iclr_generation_policy_vlm_t2i_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "Mode-window captioned image-to-image",
        [f"iclr_captioned_img2img_{s}_v1" for s in (0, 1, 2)],
    ),
]

TABLES = {
    "main_policy_repair_controls_full": {
        "rows": POLICY_REPAIR_ROWS,
        "caption": (
            "Matched feedback and repair controls on ImageNet-100-LT. All "
            "methods start from paired source checkpoints and receive three "
            "60-epoch stages with the same update cap. Generation methods add "
            "7,500 images in total. Each entry is mean $\\pm$ standard "
            "deviation over three seeds."
        ),
        "label": "tab:policy-repair-controls-full",
    },
}


def seed_result(checkpoint_root: Path, experiment: str, seed: int) -> dict | None:
    matches = sorted(checkpoint_root.glob(f"{experiment}/*/seed_{seed}/result.json"))
    for path in matches:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def collect(checkpoint_root: Path, experiments: list[str], metric: str) -> list[float]:
    """Return one value per seed, or [] if any seed is missing the metric."""
    values = []
    for seed, experiment in enumerate(experiments):
        payload = seed_result(checkpoint_root, experiment, seed)
        if payload is None:
            return []
        value = payload.get(metric)
        if not isinstance(value, (int, float)):
            return []
        values.append(float(value) * 100.0)
    return values


def cell(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"${values[0]:.2f}$"
    # Sample standard deviation, matching scripts/report_full_metric_suite.py
    # and the spread already printed in the manuscript.
    return f"${mean(values):.2f}\\pm{stdev(values):.2f}$"


def render(checkpoint_root: Path, spec: dict, protocol: str) -> tuple[str, int, int]:
    index = 1 if protocol == "linear" else 2
    header = " & ".join(name for name, _, _ in DATASETS)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{spec['caption']} "
        + (
            "Linear-probe accuracy."
            if protocol == "linear"
            else "$k$NN accuracy."
        )
        + "}",
        f"\\label{{{spec['label']}-{protocol}}}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{l" + "c" * len(DATASETS) + "}",
        "\\toprule",
        f"Method & {header} \\\\",
        "\\midrule",
    ]

    complete_rows = 0
    for label, experiments in spec["rows"]:
        cells = [
            cell(collect(checkpoint_root, experiments, dataset[index]))
            for dataset in DATASETS
        ]
        if all(value != "--" for value in cells):
            complete_rows += 1
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines), complete_rows, len(spec["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    args = parser.parse_args()

    assert len(PAPER_METRICS) == 2 * len(DATASETS), (
        "DATASETS must cover the whole paper suite"
    )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in TABLES.items():
        for protocol in ("linear", "knn"):
            body, complete, total = render(args.checkpoint_root, spec, protocol)
            path = args.tables_dir / f"{name}_{protocol}.tex"
            path.write_text(body)
            print(f"{path.name}: {complete}/{total} rows complete")


if __name__ == "__main__":
    main()
