"""Aggregate the matched closest-prior rebuttal experiments.

The report intentionally distinguishes direct source-repair baselines from
adaptations and mechanism controls.  It only compares metrics shared by every
listed experiment and reports seed counts so incomplete rows cannot silently
enter the rebuttal table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


COMMON_ACCURACIES = (
    "cars_test_accuracy",
    "aircraft_test_accuracy",
    "flowers_test_accuracy",
    "imagenet100lt_test_accuracy",
)


@dataclass(frozen=True)
class Method:
    label: str
    experiment_dirs: tuple[str, ...]
    scope: str


METHODS = (
    Method(
        "Uniform + SD3",
        ("rebuttal_factorial_uniform_sd3",),
        "Generic diffusion augmentation; matched SD3 volume",
    ),
    Method(
        "Top-tail + SD3",
        ("rebuttal_factorial_top_sd3",),
        "DOPING-inspired rare/extreme-region acquisition",
    ),
    Method(
        "Cluster-frequency + SD3",
        (
            "rebuttal_prior_cluster_inverse",
            "rebuttal_prior_cluster_inverse_gruenau_retry1",
        ),
        "Inverse cluster-occupancy acquisition",
    ),
    Method(
        "TADA-SSL + SD3",
        ("rebuttal_prior_early_loss_retry1", "rebuttal_prior_early_loss"),
        "Label-free SSL learning-difficulty adaptation of TADA",
    ),
    Method(
        "BRIDGE + SD3",
        ("rebuttal_factorial_mode_sd3",),
        "Mode-window sparse-support acquisition",
    ),
    Method(
        "BRIDGE--TADA hybrid + SD3",
        (
            "rebuttal_prior_bridge_tada",
            "rebuttal_prior_bridge_tada_gruenau_retry1",
        ),
        "Equal-budget fusion of sparse-support and SSL-difficulty acquisition",
    ),
    Method(
        "Mode-window + conventional augmentation",
        ("rebuttal_factorial_mode_strongaug",),
        "Generator-free repair control",
    ),
    Method(
        "DiffAug adaptation",
        ("rebuttal_diffaug_embedding_false",),
        "Image-SSL adaptation; not an official visual recipe",
    ),
    Method(
        "DiffAug adaptation + BRIDGE",
        ("rebuttal_diffaug_embedding_true",),
        "Image-SSL adaptation with targeted SD3 repair",
    ),
    Method(
        "SD3 feature distillation",
        ("rebuttal_sd3_vae_distill_false",),
        "External diffusion-prior control without generated images",
    ),
    Method(
        "SD3 feature distillation + BRIDGE",
        ("rebuttal_sd3_vae_distill_true",),
        "External diffusion prior plus targeted source repair",
    ),
)


def finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def find_results(roots: list[Path], names: tuple[str, ...]) -> list[Path]:
    by_seed: dict[int, Path] = {}
    for root in roots:
        for name in names:
            experiment_root = root / name
            if not experiment_root.exists():
                continue
            for result in sorted(experiment_root.rglob("result.json")):
                seed_parts = [
                    part for part in result.parts if part.startswith("seed_")
                ]
                if not seed_parts:
                    continue
                try:
                    seed = int(seed_parts[-1].split("_", 1)[1])
                except ValueError:
                    continue
                by_seed.setdefault(seed, result)
    return [by_seed[seed] for seed in sorted(by_seed)]


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_percent(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "--"
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    roots = [Path(root) for root in args.root]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for method in METHODS:
        result_paths = find_results(roots, method.experiment_dirs)
        seed_rows = []
        for result_path in result_paths:
            result = json.loads(result_path.read_text())
            common = [finite_float(result.get(key)) for key in COMMON_ACCURACIES]
            if any(value is None for value in common):
                continue
            seed_rows.append(
                {
                    "path": str(result_path),
                    "common_four_average": statistics.mean(common),
                    "knn_1_accuracy": finite_float(
                        result.get("representation/knn_1_accuracy")
                    ),
                    "training_time_hours": finite_float(
                        result.get("training_time")
                    ),
                }
            )

        common_mean, common_std = mean_std(
            [float(row["common_four_average"]) for row in seed_rows]
        )
        knn_mean, knn_std = mean_std(
            [
                float(row["knn_1_accuracy"])
                for row in seed_rows
                if row["knn_1_accuracy"] is not None
            ]
        )
        hours_mean, hours_std = mean_std(
            [
                float(row["training_time_hours"])
                for row in seed_rows
                if row["training_time_hours"] is not None
            ]
        )
        rows.append(
            {
                "method": method.label,
                "scope": method.scope,
                "num_seeds": len(seed_rows),
                "complete": len(seed_rows) == args.expected_seeds,
                "common_four_mean": common_mean,
                "common_four_std": common_std,
                "knn_mean": knn_mean,
                "knn_std": knn_std,
                "hours_mean": hours_mean,
                "hours_std": hours_std,
                "seed_results": seed_rows,
            }
        )

    (output / "closest_prior_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    markdown = [
        "# Closest-prior diffusion augmentation comparison",
        "",
        "All accuracy entries are mean ± sample standard deviation over paired "
        "seeds. The transfer average uses Cars, Aircraft, Flowers, and "
        "ImageNet-100-LT, which are available for every method.",
        "",
        "| Method | Seeds | Four-task transfer (%) | Representation 1-NN (%) | "
        "GPU-hours | Scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        hours = (
            "--"
            if row["hours_mean"] is None
            else f"{row['hours_mean']:.2f} ± {row['hours_std']:.2f}"
        )
        markdown.append(
            f"| {row['method']} | {row['num_seeds']} | "
            f"{format_percent(row['common_four_mean'], row['common_four_std'])} | "
            f"{format_percent(row['knn_mean'], row['knn_std'])} | {hours} | "
            f"{row['scope']} |"
        )
    (output / "closest_prior_table.md").write_text("\n".join(markdown) + "\n")

    latex = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Seeds & Four-task transfer & Representation 1-NN \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = str(row["method"]).replace("&", "\\&")
        latex.append(
            f"{label} & {row['num_seeds']} & "
            f"{format_percent(row['common_four_mean'], row['common_four_std'])} & "
            f"{format_percent(row['knn_mean'], row['knn_std'])} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "closest_prior_table.tex").write_text("\n".join(latex) + "\n")

    incomplete = [row["method"] for row in rows if not row["complete"]]
    if incomplete:
        print("Incomplete methods:", ", ".join(map(str, incomplete)))
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
