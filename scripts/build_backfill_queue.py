#!/usr/bin/env python3
"""Build a prioritized, self-describing queue of full-metric backfill cells.

Every queued cell carries the model/ssl/dataset configuration recovered from
the protocol file that was written next to its checkpoint.  Guessing those
values is not acceptable: loading a ViT-S checkpoint under the default
ResNet-50 config would silently produce a meaningless result rather than an
error, and that result would then flow into a paper table.

The queue is a directory of one file per cell so that independent workers on
different clusters can claim work atomically with ``mv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiment_inventory import missing_metrics

MODEL_CONFIGS = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "ResNet101": "resnet101",
    "ViTTiny": "vit_tiny",
    "ViTSmall": "vit_small",
    "ViTBase": "vit_base",
}

SSL_CONFIGS = {
    "SimCLR": "simclr",
    "MoCo": "moco",
    "Dino": "dino",
    "MAE": "mae",
    "SDCLR": "sdclr",
    "DiffAug": "diffaug",
    "Supervised": "supervised",
}

# Only imbalanced source regimes appear in the paper's result tables.
DATASET_CONFIGS = {
    "clane9/imagenet-100": "imagenet100_imbalanced",
    "uoft-cs/cifar10": "cifar10_imbalanced",
    "uoft-cs/cifar100": "cifar100_imbalanced",
    "poloclub/diffusiondb": "diffusiondb_subset",
    "experiment/dataset/hf_scripts/pass_subset.py": "pass_subset",
}

# Priority 1 feeds the main tables, priority 2 the appendix robustness and
# scaling tables.  Anything not listed is screening or smoke-test work that
# never reaches the paper and is not queued at all.
PRIORITY_PREFIXES = [
    (
        "10",
        (
            "rebuttal_fullpolicy3e60_",
            "iclr_aide_",
            "iclr_captioned_",
            "iclr_generation_",
        ),
    ),
    ("11", ("rebuttal_full5e100_", "iclr_fullpolicy5e100_")),
    ("12", ("rebuttal_percentile_",)),
    ("13", ("rebuttal_feedback_", "rebuttal_adaptive_")),
    ("14", ("rebuttal_compat_", "rebuttal_factorial_")),
    # The generator sweeps are what let the paper withdraw the "SD3 beats
    # FLUX" claim: the comparison is only meaningful with both sides measured
    # under matched sampling settings.
    ("15", ("rebuttal_generator_", "rebuttal_flux_")),
    ("16", ("rebuttal_sd3_", "rebuttal_diffaug_", "rebuttal_causal_geometry_")),
    ("20", ("rebuttal_selector_",)),
    ("21", ("rebuttal_scale_",)),
    ("22", ("rebuttal_cycles_", "rebuttal_ts_")),
]


# Run-name suffixes that distinguish reruns of the same condition.  Two names
# that agree once these are stripped denote the same experimental cell, so
# only one cluster should spend a GPU on it.
RERUN_SUFFIXES = (
    "_iclr2027_full_v2",
    "_iclr2027_full_retry1",
    "_gruenau_retry1",
    "_gruenau_retry3",
    "_iclr2027",
    "_retry1",
    "_recovery",
    "_clean",
    "_v1",
    "_v2",
)


def canonical_family(experiment: str) -> str:
    name = experiment
    for suffix in RERUN_SUFFIXES:
        name = name.replace(suffix, "")
    return name


def priority_for(experiment: str) -> str | None:
    for priority, prefixes in PRIORITY_PREFIXES:
        if experiment.startswith(prefixes):
            return priority
    return None


def describe(seed_dir: Path) -> dict | None:
    """Recover the launch configuration recorded beside a checkpoint."""
    for name in ("protocol.json", "metric_backfill_protocol.json"):
        protocol_path = seed_dir / name
        if not protocol_path.is_file():
            continue
        try:
            protocol = json.loads(protocol_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        model_name = protocol.get("model", {}).get("model_name")
        ssl_method = protocol.get("ssl", {}).get("ssl_method")
        dataset_path = protocol.get("dataset", {}).get("dataset_path")
        if model_name in MODEL_CONFIGS and ssl_method in SSL_CONFIGS:
            if dataset_path in DATASET_CONFIGS:
                return {
                    "model_config": MODEL_CONFIGS[model_name],
                    "ssl_config": SSL_CONFIGS[ssl_method],
                    "dataset_config": DATASET_CONFIGS[dataset_path],
                }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the plan without writing"
    )
    parser.add_argument(
        "--claimed-elsewhere",
        type=Path,
        help=(
            "File listing experiment names already held by another cluster. "
            "Cells whose canonical family appears there are not queued, so "
            "two clusters never evaluate the same condition twice."
        ),
    )
    args = parser.parse_args()

    claimed: set[str] = set()
    if args.claimed_elsewhere:
        claimed = {
            canonical_family(line.strip())
            for line in args.claimed_elsewhere.read_text().splitlines()
            if line.strip()
        }

    pending = args.queue / "pending"
    if not args.dry_run:
        for state in ("pending", "running", "completed", "failed", "logs"):
            (args.queue / state).mkdir(parents=True, exist_ok=True)

    queued = 0
    skipped_priority = 0
    skipped_claimed = 0
    undescribed = []
    by_priority: dict[str, int] = {}

    for result_path in sorted(args.checkpoint_root.glob("*/*/seed_*/result.json")):
        seed_dir = result_path.parent
        parts = result_path.relative_to(args.checkpoint_root).parts
        experiment, dataset_dir = parts[0], parts[1]
        seed = seed_dir.name.removeprefix("seed_")

        absent = missing_metrics(result_path)
        if not absent:
            continue
        if not (seed_dir / "last.ckpt").is_file():
            continue

        priority = priority_for(experiment)
        if priority is None:
            skipped_priority += 1
            continue
        if canonical_family(experiment) in claimed:
            skipped_claimed += 1
            continue

        spec = describe(seed_dir)
        if spec is None:
            undescribed.append(f"{experiment} seed {seed}")
            continue

        cell = {
            "experiment": experiment,
            "seed": seed,
            "dataset_dir": dataset_dir,
            "missing_metrics": len(absent),
            **spec,
        }
        by_priority[priority] = by_priority.get(priority, 0) + 1
        queued += 1

        if not args.dry_run:
            name = f"{priority}__{experiment}__seed{seed}.json"
            # Claimed cells are renamed to "<name>.<worker>", so an exact-name
            # lookup would miss in-flight work and queue it a second time.
            # Two workers writing one result.json concurrently would corrupt
            # it, so match any state file that starts with the cell name.
            if not any(
                any((args.queue / state).glob(f"{name}*"))
                for state in ("pending", "running", "completed", "failed")
            ):
                (pending / name).write_text(json.dumps(cell, indent=2))

    print(f"queued cells:            {queued}")
    for priority in sorted(by_priority):
        print(f"  priority {priority}: {by_priority[priority]}")
    print(f"skipped (not in paper):  {skipped_priority}")
    print(f"skipped (other cluster): {skipped_claimed}")
    if undescribed:
        print(f"\nUNRESOLVED CONFIG ({len(undescribed)}) - not queued:")
        for item in undescribed:
            print(f"  {item}")


if __name__ == "__main__":
    main()
