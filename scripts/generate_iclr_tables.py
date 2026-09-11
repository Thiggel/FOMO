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
# Matched three-stage controls, one directory per seed.  Every acquisition and
# repair condition here is a post-a29d244 run: before that commit the checkpoint
# callback skipped a save whenever the step counter matched one it had already
# written, which is every cycle after the first, so each condition published its
# first-cycle encoder and the adaptive and frozen arms came out bit-identical.
# The three generation-policy rows below are named separately because two of
# them are still waiting on reruns.
POLICY_REPAIR_ROWS = [
    (
        "Adaptive \\method",
        [
            "rebuttal_fullpolicy3e60_adaptive_0_sep10",
            "rebuttal_fullpolicy3e60_adaptive_1_fix2",
            "rebuttal_fullpolicy3e60_adaptive_2_sep10",
        ],
    ),
    (
        "Frozen first-cycle selector",
        [f"rebuttal_fullpolicy3e60_static_{s}_fix2" for s in (0, 1, 2)],
    ),
    (
        "One-shot repair",
        [f"rebuttal_fullpolicy3e60_one_shot_{s}_ckptfix" for s in (0, 1, 2)],
    ),
    (
        "No repair",
        [f"rebuttal_fullpolicy3e60_no_repair_{s}_sep10" for s in (0, 1, 2)],
    ),
    (
        "Mode-window conventional augmentation",
        [
            "rebuttal_fullpolicy3e60_conventional_0_fix2",
            "rebuttal_fullpolicy3e60_conventional_1_sep10",
            "rebuttal_fullpolicy3e60_conventional_2_sep10",
        ],
    ),
    (
        "Uniform acquisition",
        [f"rebuttal_fullpolicy3e60_uniform_{s}_fix2" for s in (0, 1, 2)],
    ),
    (
        "Top-tail acquisition",
        [
            "rebuttal_fullpolicy3e60_top_tail_0_fix2",
            "rebuttal_fullpolicy3e60_top_tail_1_fix2",
            "rebuttal_fullpolicy3e60_top_tail_2_sep10",
        ],
    ),
    (
        "AIDE-style VLM acquisition and text-to-image",
        [f"iclr_aide_ssl_clip_cluster_{s}_ckptfix" for s in (0, 1, 2)],
    ),
    # Awaiting reruns.  Seeds 1 and 2 of the captioned image-to-image arm and
    # all three of the text-to-image arm still resolve to pre-fix directories,
    # so these two rows are left pointing at runs that will be replaced.
    (
        "Mode-window caption and text-to-image",
        [f"iclr_generation_policy_vlm_t2i_{s}_iclr2027" for s in (0, 1, 2)],
    ),
    (
        "Mode-window captioned image-to-image",
        [
            "iclr_captioned_img2img_0_ckptfix",
            "iclr_captioned_img2img_1_v1",
            "iclr_captioned_img2img_2_v1",
        ],
    ),
]

# Paired full-schedule ViT-S runs, one directory per seed.  Names are spelled
# out because the cells were finished across four waves: the ckptfix runs of
# late August, the two objective waves that carry no suffix, and the sep05 wave
# that replaced every cell still standing on a pre-fix encoder.  Each entry is
# the newest run of that cell holding all fourteen paper metrics.
#
# Every cell here postdates commit a29d244.  Before it, Lightning skipped the
# save whenever ``_last_global_step_saved`` equalled ``global_step``, which is
# the state every cycle after the first arrives in, so a five-cycle run
# published its first-cycle encoder.  A pre-fix repair arm therefore understates
# the method rather than flattering it.
GENERALIZATION_ROWS = [
    (
        "SimCLR, SSL baseline",
        [
            "rebuttal_full5e100_simclr_vits_base_0_ckptfix2",
            "rebuttal_full5e100_simclr_vits_base_1_ckptfix",
            "rebuttal_full5e100_simclr_vits_base_2_ckptfix2",
        ],
    ),
    (
        "SimCLR, with \\method",
        [
            "rebuttal_full5e100_simclr_vits_bridge_0_sep05",
            "rebuttal_full5e100_simclr_vits_bridge_1_ckptfix",
            "rebuttal_full5e100_simclr_vits_bridge_2_sep05",
        ],
    ),
    (
        "MoCo v3, SSL baseline",
        [f"rebuttal_full5e100_mocov3_base_{s}_ckptfix" for s in (0, 1, 2)],
    ),
    (
        "MoCo v3, with \\method",
        [
            "rebuttal_full5e100_mocov3_bridge_0_ckptfix",
            "rebuttal_full5e100_mocov3_bridge_1_sep05",
            "rebuttal_full5e100_mocov3_bridge_2_ckptfix",
        ],
    ),
    (
        "DINO, SSL baseline",
        [f"rebuttal_full5e100_dino_base_{s}_ckptfix2" for s in (0, 1, 2)],
    ),
    (
        "DINO, with \\method",
        [
            "rebuttal_full5e100_dino_bridge_0_ckptfix2",
            "rebuttal_full5e100_dino_bridge_1_sep05",
            "rebuttal_full5e100_dino_bridge_2_sep05",
        ],
    ),
    (
        "DINOv2, SSL baseline",
        [
            "rebuttal_full5e80_dinov2_base_0_sep05",
            "rebuttal_full5e80_dinov2_base_1",
            "rebuttal_full5e80_dinov2_base_2",
        ],
    ),
    (
        "DINOv2, with \\method",
        [
            "rebuttal_full5e80_dinov2_bridge_0",
            "rebuttal_full5e80_dinov2_bridge_1",
            "rebuttal_full5e80_dinov2_bridge_2_sep05",
        ],
    ),
    (
        "MAE, SSL baseline",
        [
            "rebuttal_full5e80_mae_base_0",
            "rebuttal_full5e80_mae_base_1_sep05",
            "rebuttal_full5e80_mae_base_2_sep05",
        ],
    ),
    (
        "MAE, with \\method",
        [
            "rebuttal_full5e80_mae_bridge_0",
            "rebuttal_full5e80_mae_bridge_1_sep05",
            "rebuttal_full5e80_mae_bridge_2_sep05",
        ],
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
    # Cosine scoring divides the normalized-L2 distance by a positive
    # constant, which cannot change a ranking or a quantile position, so this
    # row selects exactly the same anchors as the normalized-L2 row above.  It
    # is kept deliberately: two rows that are provably the same experiment
    # measure the run-to-run spread of the whole pipeline, which calibrates
    # how large a difference elsewhere in this table is readable.
    ("Distance: cosine (null control)", "rebuttal_selector_metric_cosine_iclr2027_full_v2"),
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
            "standard deviation over three seeds. Cosine scoring rescales the "
            "normalized $L_2$ score by a positive constant and therefore "
            "selects the identical anchor set; that row is a null control, and "
            "its distance from the normalized $L_2$ row measures the "
            "end-to-end run-to-run spread of the pipeline."
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
            "number of post-branch optimizer updates. Five objectives span "
            "contrastive (SimCLR, MoCo v3), self-distillation (DINO, "
            "DINOv2) and masked-reconstruction (MAE) pretraining. SimCLR, "
            "MoCo v3 and DINO run five cycles of 100 epochs; DINOv2 and MAE "
            "run five of 80, since DINOv2 carries multi-crop at batch 16 and "
            "does not finish otherwise. Mean $\\pm$ standard deviation over "
            "three seeds."
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


def training_completed(result_path: Path) -> bool:
    """Whether the run behind a result finished all of its repair cycles.

    ``last.ckpt`` is rewritten after every completed cycle, so a run that dies
    partway leaves the encoder from an earlier cycle in place, and the metric
    backfill scores it as though it were final.  Conditions that only diverge
    in later cycles then share one encoder while still printing distinct
    linear-probe numbers, because the probe is stochastic.  Runs that predate
    the progress marker have no file; treat those as unknown rather than
    complete, so they have to be re-run or explicitly waived.
    """
    progress = result_path.parent / "training_progress.json"
    try:
        payload = json.loads(progress.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("complete"))


def seed_result(
    checkpoint_root: Path,
    experiment: str,
    seed: int,
    require_complete: bool = False,
) -> dict | None:
    matches = sorted(checkpoint_root.glob(f"{experiment}/*/seed_{seed}/result.json"))
    for path in matches:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if require_complete and not training_completed(path):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def collect(
    checkpoint_root: Path,
    experiments: list[str],
    metric: str,
    require_complete: bool = False,
) -> list[float]:
    """Return one value per seed, or [] if any seed is missing the metric."""
    values = []
    for seed, experiment in enumerate(experiments):
        payload = seed_result(
            checkpoint_root, experiment, seed, require_complete=require_complete
        )
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


def render(
    checkpoint_root: Path,
    spec: dict,
    protocol: str,
    require_complete: bool = False,
) -> tuple[str, int, int]:
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
            cell(
                collect(
                    checkpoint_root,
                    experiments,
                    dataset[index],
                    require_complete=require_complete,
                )
            )
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


def duplicate_rows(
    checkpoint_root: Path,
    spec: dict,
    protocol: str,
    require_complete: bool = False,
) -> list[str]:
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
            tuple(
                collect(
                    checkpoint_root,
                    experiments,
                    dataset[index],
                    require_complete=require_complete,
                )
            )
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
    parser.add_argument(
        "--require-complete-cycles",
        action="store_true",
        help=(
            "Only use results whose run recorded finishing every repair "
            "cycle.  Runs from before the progress marker existed count as "
            "incomplete, so this blanks any row not yet re-run."
        ),
    )
    args = parser.parse_args()

    assert len(PAPER_METRICS) == 2 * len(DATASETS), (
        "DATASETS must cover the whole paper suite"
    )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in TABLES.items():
        for protocol in ("linear", "knn"):
            body, complete, total = render(
                args.checkpoint_root,
                spec,
                protocol,
                require_complete=args.require_complete_cycles,
            )
            path = args.tables_dir / f"{name}_{protocol}.tex"
            path.write_text(body)
            print(f"{path.name}: {complete}/{total} rows complete")
            for collision in duplicate_rows(
                args.checkpoint_root,
                spec,
                protocol,
                require_complete=args.require_complete_cycles,
            ):
                print(f"  WARNING identical on every dataset: {collision}")


if __name__ == "__main__":
    main()
