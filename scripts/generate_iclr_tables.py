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

# Paired full-schedule ViT-S runs.  mocov3_bridge seed 1 was attempted three
# times; retry1 stopped before writing a result and retry2 produced nothing,
# so retry3 is the only completed run and is the one reported here.
GENERALIZATION_ROWS = [
    (
        "SimCLR, SSL baseline",
        [f"rebuttal_full5e100_simclr_vits_base_{s}_gruenau_retry1" for s in (0, 1, 2)],
    ),
    (
        "SimCLR, with \\method",
        [
            f"rebuttal_full5e100_simclr_vits_bridge_{s}_gruenau_retry1"
            for s in (0, 1, 2)
        ],
    ),
    (
        "MoCo v3, SSL baseline",
        [f"rebuttal_full5e100_mocov3_base_{s}_gruenau_retry1" for s in (0, 1, 2)],
    ),
    (
        "MoCo v3, with \\method",
        [
            "rebuttal_full5e100_mocov3_bridge_0_gruenau_retry1",
            "rebuttal_full5e100_mocov3_bridge_1_gruenau_retry3",
            "rebuttal_full5e100_mocov3_bridge_2_gruenau_retry1",
        ],
    ),
    # DINO completed only two seeds and the paper reports it as inconclusive
    # rather than as evidence for or against the method.
    (
        "DINO, SSL baseline",
        [f"rebuttal_full5e100_dino_base_{s}_gruenau_retry1" for s in (0, 1)],
    ),
    (
        "DINO, with \\method",
        [f"rebuttal_full5e100_dino_bridge_{s}_gruenau_retry1" for s in (0, 1)],
    ),
]

# Sparsity bands, from the densest quarter of the score distribution to the
# extreme tail.  Proposition 3 predicts utility peaks at an interior band
# rather than at either end, so the ordering across these rows is the
# empirical test of the interior-optimum claim.  Each band is one experiment
# directory holding all three seeds.
PERCENTILE_BANDS = [
    ("q0--25 (densest)", "rebuttal_percentile_q00_25_iclr2027_full_v2"),
    ("q25--50", "rebuttal_percentile_q25_50_iclr2027_full_v2"),
    ("q50--75", "rebuttal_percentile_q50_75_iclr2027_full_v2"),
    ("q75--85", "rebuttal_percentile_q75_85_iclr2027_full_v2"),
    ("q85--90", "rebuttal_percentile_q85_90_iclr2027_full_v2"),
    ("q90--95", "rebuttal_percentile_q90_95_iclr2027_full_v2"),
    ("q95--97", "rebuttal_percentile_q95_97_iclr2027_full_v2"),
    ("q97--99", "rebuttal_percentile_q97_99_iclr2027_full_v2"),
    ("q99--100 (extreme tail)", "rebuttal_percentile_q99_100_iclr2027_full_v2"),
]

PERCENTILE_ROWS = [(label, [name] * 3) for label, name in PERCENTILE_BANDS]

# Selector robustness: one row per setting, grouped by the knob it varies.
# Most conditions were rerun under the iclr2027_full_v2 wave; alpha4 exists
# only from the earlier gruenau retry, so it is named explicitly rather than
# resolved by a fallback rule that could silently mix waves.  cutoff100 and k10
# were in that same position until their v2 seeds landed; their v2 protocols
# were checked field for field against the retry runs before repointing, and
# v2 carries all three seeds where the retry had two and one.
SELECTOR_SETTINGS = [
    ("Candidate pool $\\alpha=1$", "rebuttal_selector_alpha1_iclr2027_full_v2"),
    ("Candidate pool $\\alpha=2$", "rebuttal_selector_alpha2_iclr2027_full_v2"),
    ("Candidate pool $\\alpha=4$ (default)", "rebuttal_selector_alpha4_gruenau_retry1"),
    ("Candidate pool $\\alpha=8$", "rebuttal_selector_alpha8_iclr2027_full_v2"),
    ("Upper cutoff $q=0.95$", "rebuttal_selector_cutoff95_iclr2027_full_v2"),
    ("Upper cutoff $q=0.99$ (default)", "rebuttal_selector_cutoff99_iclr2027_full_v2"),
    ("Upper cutoff $q=0.995$", "rebuttal_selector_cutoff995_iclr2027_full_v2"),
    ("Upper cutoff $q=1.0$", "rebuttal_selector_cutoff100_iclr2027_full_v2"),
    ("Neighbourhood $k=10$", "rebuttal_selector_k10_iclr2027_full_v2"),
    ("Neighbourhood $k=25$", "rebuttal_selector_k25_iclr2027_full_v2"),
    ("Neighbourhood $k=50$", "rebuttal_selector_k50_iclr2027_full_v2"),
    ("Neighbourhood $k=200$", "rebuttal_selector_k200_iclr2027_full_v2"),
    ("Distance: normalized $L_2$ (default)", "rebuttal_selector_metric_normalized_iclr2027_full_v2"),
    ("Distance: cosine", "rebuttal_selector_metric_cosine_iclr2027_full_v2"),
    ("Distance: raw $L_2$", "rebuttal_selector_metric_raw_iclr2027_full_v2"),
    ("Strategy: mode window (default)", "rebuttal_selector_strategy_mode_iclr2027_full_v2"),
    ("Strategy: sparse band with FPS", "rebuttal_selector_strategy_band_fps_iclr2027_full_v2"),
    ("Strategy: sparse band, random", "rebuttal_selector_strategy_band_random_iclr2027_full_v2"),
    ("Strategy: inverse-cluster", "rebuttal_selector_strategy_cluster_iclr2027_full_v2"),
    ("Strategy: densest window", "rebuttal_selector_strategy_densest_iclr2027_full_v2"),
    ("Strategy: top tail, no diversity", "rebuttal_selector_strategy_top_iclr2027_full_v2"),
]

SELECTOR_ROWS = [(label, [name] * 3) for label, name in SELECTOR_SETTINGS]

TABLES = {
    "app_selector_robustness_full": {
        "rows": SELECTOR_ROWS,
        "caption": (
            "Selector robustness on ImageNet-100-LT. Each row changes one "
            "component of the acquisition rule -- candidate pool multiplier, "
            "upper cutoff, neighbourhood size, distance, or selection "
            "strategy -- and holds the rest of the loop fixed. Mean $\\pm$ "
            "standard deviation over three seeds."
        ),
        "label": "tab:selector-robustness-full",
    },
    "main_percentile_utility_full": {
        "rows": PERCENTILE_ROWS,
        "caption": (
            "Repair utility by sparsity band on ImageNet-100-LT. Each row "
            "spends the identical acquisition budget on a different quantile "
            "range of the $k$NN score distribution, holding every other part "
            "of the loop fixed. Mean $\\pm$ standard deviation over three "
            "seeds."
        ),
        "label": "tab:percentile-utility-full",
    },
    "main_generalization_vits_full": {
        "rows": GENERALIZATION_ROWS,
        "caption": (
            "Paired full-schedule ViT-S transfer on ImageNet-100-LT, with "
            "every downstream dataset reported separately. Both arms of each "
            "pair start from the same source checkpoint and receive the same "
            "number of post-branch optimizer updates."
        ),
        "label": "tab:generalization-vits-full",
    },
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
        # Never let a row with fewer seeds pass as one of the standard three.
        shown = label if len(experiments) == 3 else (
            f"{label} ({len(experiments)} seeds)"
        )
        lines.append(f"{shown} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines), complete_rows, len(spec["rows"])


def duplicate_rows(checkpoint_root: Path, spec: dict, protocol: str) -> list[str]:
    """Report rows whose numbers are identical across every dataset.

    Two conditions that agree to the last digit on all seven datasets did not
    merely perform similarly, they are the same model: this is how the frozen
    first-cycle selector was found to produce an encoder bit-identical to
    adaptive selection, which made the paper's adaptive-versus-frozen
    comparison a model compared against itself.
    """
    index = 1 if protocol == "linear" else 2
    fingerprints: dict[tuple, list[str]] = {}
    for label, experiments in spec["rows"]:
        values = tuple(
            tuple(collect(checkpoint_root, experiments, dataset[index]))
            for dataset in DATASETS
        )
        if all(values):
            fingerprints.setdefault(values, []).append(label)
    return [
        " == ".join(labels) for labels in fingerprints.values() if len(labels) > 1
    ]


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
            for collision in duplicate_rows(args.checkpoint_root, spec, protocol):
                print(f"  WARNING identical on every dataset: {collision}")


if __name__ == "__main__":
    main()
