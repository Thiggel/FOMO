"""Statistical audit of completed rebuttal experiments.

The primary analysis is a two-sided paired t test over matched training seeds.
The report also includes a 95 percent t confidence interval for the paired
difference, Cohen's dz, an exact two-sided Wilcoxon test where SciPy permits it,
and Holm-adjusted p values within each experimental family.

Only independent training seeds are treated as replicates. Downstream datasets,
classes, checkpoints, and generated images are never treated as independent
replicates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats


Metric = Callable[[dict], float]


ROOT = Path("/vol/home-vol2/ml/laitenbf/FOMO_runtime/checkpoints")
REPO_ROOT = Path(__file__).resolve().parents[2]
FOUR_TASK_KEYS = (
    "cars_test_accuracy",
    "aircraft_test_accuracy",
    "flowers_test_accuracy",
    "imagenet100lt_test_accuracy",
)


def accuracy(key: str) -> Metric:
    return lambda row: 100.0 * float(row[key])


def raw(key: str) -> Metric:
    return lambda row: float(row[key])


def four_task_average(row: dict) -> float:
    return 25.0 * sum(float(row[key]) for key in FOUR_TASK_KEYS)


def load_seed_map(directory: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in directory.rglob("result.json"):
        name = path.parent.name
        if not name.startswith("seed_"):
            continue
        try:
            seed = int(name.split("_", 1)[1])
            payload = json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
        rows[seed] = payload
    return rows


def load_named(name: str) -> dict[int, dict]:
    return load_seed_map(ROOT / name)


def load_per_seed(names: dict[int, str]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for seed, name in names.items():
        rows = load_named(name)
        if seed in rows:
            result[seed] = rows[seed]
    return result


@dataclass
class Comparison:
    family: str
    label: str
    control: dict[int, dict]
    treatment: dict[int, dict]
    metric_name: str
    metric: Metric


@dataclass
class SummaryComparison:
    family: str
    label: str
    metric: str
    control_mean: float
    control_sd: float
    treatment_mean: float
    treatment_sd: float
    n_control: int = 3
    n_treatment: int = 3


def holm_adjust(p_values: list[float]) -> list[float]:
    result = [math.nan] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid):
        adjusted = min(1.0, (total - rank) * value)
        running = max(running, adjusted)
        result[index] = running
    return result


def paired_result(comparison: Comparison) -> dict:
    seeds = sorted(set(comparison.control) & set(comparison.treatment))
    control = np.asarray([comparison.metric(comparison.control[s]) for s in seeds], dtype=float)
    treatment = np.asarray([comparison.metric(comparison.treatment[s]) for s in seeds], dtype=float)
    difference = treatment - control
    n = len(seeds)

    output = {
        "family": comparison.family,
        "comparison": comparison.label,
        "metric": comparison.metric_name,
        "seeds": seeds,
        "n": n,
        "control_values": control.tolist(),
        "treatment_values": treatment.tolist(),
        "difference_values": difference.tolist(),
    }
    if n == 0:
        return output

    output.update(
        {
            "control_mean": float(control.mean()),
            "treatment_mean": float(treatment.mean()),
            "difference_mean": float(difference.mean()),
            "control_sd": float(control.std(ddof=1)) if n > 1 else math.nan,
            "treatment_sd": float(treatment.std(ddof=1)) if n > 1 else math.nan,
            "difference_sd": float(difference.std(ddof=1)) if n > 1 else math.nan,
        }
    )
    if n < 2:
        return output

    standard_error = float(stats.sem(difference))
    critical = float(stats.t.ppf(0.975, n - 1))
    output["ci95_low"] = float(difference.mean() - critical * standard_error)
    output["ci95_high"] = float(difference.mean() + critical * standard_error)
    test = stats.ttest_rel(treatment, control)
    output["paired_t"] = float(test.statistic)
    output["paired_t_p_two_sided"] = float(test.pvalue)
    difference_sd = float(difference.std(ddof=1))
    output["cohen_dz"] = float(difference.mean() / difference_sd) if difference_sd else math.inf
    try:
        wilcoxon = stats.wilcoxon(treatment, control, alternative="two-sided", method="auto")
        output["wilcoxon_p_two_sided"] = float(wilcoxon.pvalue)
    except ValueError:
        output["wilcoxon_p_two_sided"] = math.nan
    return output


def summary_result(comparison: SummaryComparison) -> dict:
    test = stats.ttest_ind_from_stats(
        mean1=comparison.treatment_mean,
        std1=comparison.treatment_sd,
        nobs1=comparison.n_treatment,
        mean2=comparison.control_mean,
        std2=comparison.control_sd,
        nobs2=comparison.n_control,
        equal_var=False,
    )
    variance = (
        comparison.treatment_sd**2 / comparison.n_treatment
        + comparison.control_sd**2 / comparison.n_control
    )
    standard_error = math.sqrt(variance)
    numerator = variance**2
    denominator = (
        (comparison.treatment_sd**2 / comparison.n_treatment) ** 2
        / (comparison.n_treatment - 1)
        + (comparison.control_sd**2 / comparison.n_control) ** 2
        / (comparison.n_control - 1)
    )
    degrees_freedom = numerator / denominator
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    difference = comparison.treatment_mean - comparison.control_mean
    pooled_variance = (
        (comparison.n_treatment - 1) * comparison.treatment_sd**2
        + (comparison.n_control - 1) * comparison.control_sd**2
    ) / (comparison.n_treatment + comparison.n_control - 2)
    pooled_sd = math.sqrt(pooled_variance)
    return {
        "family": comparison.family,
        "comparison": comparison.label,
        "metric": comparison.metric,
        "n_control": comparison.n_control,
        "n_treatment": comparison.n_treatment,
        "control_mean": comparison.control_mean,
        "control_sd": comparison.control_sd,
        "treatment_mean": comparison.treatment_mean,
        "treatment_sd": comparison.treatment_sd,
        "difference_mean": difference,
        "ci95_low": difference - critical * standard_error,
        "ci95_high": difference + critical * standard_error,
        "welch_t": float(test.statistic),
        "welch_df": degrees_freedom,
        "welch_p_two_sided": float(test.pvalue),
        "cohen_d": difference / pooled_sd if pooled_sd else math.inf,
    }


def comparisons() -> list[Comparison]:
    full_moco_base = load_per_seed(
        {
            0: "rebuttal_full5e100_mocov3_base_0_gruenau_retry1",
            1: "rebuttal_full5e100_mocov3_base_1_gruenau_retry1",
            2: "rebuttal_full5e100_mocov3_base_2_gruenau_retry1",
        }
    )
    full_moco_bridge = load_per_seed(
        {
            0: "rebuttal_full5e100_mocov3_bridge_0_gruenau_retry1",
            1: "rebuttal_full5e100_mocov3_bridge_1_gruenau_retry3",
            2: "rebuttal_full5e100_mocov3_bridge_2_gruenau_retry1",
        }
    )

    result: list[Comparison] = []
    for metric_name, metric in (
        ("four task average", four_task_average),
        ("Cars accuracy", accuracy("cars_test_accuracy")),
        ("Aircraft accuracy", accuracy("aircraft_test_accuracy")),
        ("Flowers accuracy", accuracy("flowers_test_accuracy")),
        ("ImageNet 100 LT accuracy", accuracy("imagenet100lt_test_accuracy")),
        ("effective rank", raw("representation/effective_rank")),
        ("spectral entropy", raw("representation/spectral_entropy")),
        ("1 NN accuracy", accuracy("representation/knn_1_accuracy")),
    ):
        result.append(
            Comparison(
                "full MoCo v3 ViT S",
                "BRIDGE minus MoCo v3",
                full_moco_base,
                full_moco_bridge,
                metric_name,
                metric,
            )
        )

    for family_name, directory_family, seeds in (
        ("full SimCLR ViT S", "simclr_vits", (0, 1, 2)),
        ("full DINO ViT S", "dino", (0, 1)),
    ):
        base = load_per_seed(
            {
                seed: f"rebuttal_full5e100_{directory_family}_base_{seed}_gruenau_retry1"
                for seed in seeds
            }
        )
        bridge = load_per_seed(
            {
                seed: f"rebuttal_full5e100_{directory_family}_bridge_{seed}_gruenau_retry1"
                for seed in seeds
            }
        )
        for metric_name, metric in (
            ("four task average", four_task_average),
            ("Cars accuracy", accuracy("cars_test_accuracy")),
            ("Aircraft accuracy", accuracy("aircraft_test_accuracy")),
            ("Flowers accuracy", accuracy("flowers_test_accuracy")),
            ("ImageNet 100 LT accuracy", accuracy("imagenet100lt_test_accuracy")),
            ("effective rank", raw("representation/effective_rank")),
            ("spectral entropy", raw("representation/spectral_entropy")),
            ("1 NN accuracy", accuracy("representation/knn_1_accuracy")),
        ):
            result.append(
                Comparison(
                    family_name,
                    f"BRIDGE minus {family_name.removeprefix('full ')}",
                    base,
                    bridge,
                    metric_name,
                    metric,
                )
            )

    result.extend(
        [
            Comparison(
                "short ViT diagnostics",
                "one cycle MoCo BRIDGE minus base",
                load_named("rebuttal_compat_mocov3_vits_base"),
                load_named("rebuttal_compat_mocov3_vits_bridge"),
                "four task average",
                four_task_average,
            ),
            Comparison(
                "short ViT diagnostics",
                "three cycle MoCo BRIDGE minus base",
                load_per_seed(
                    {i: f"deadline_mc3e60_mocov3_base_{i}_gruenau_retry1" for i in range(3)}
                ),
                load_per_seed(
                    {i: f"deadline_mc3e60_mocov3_bridge_{i}_gruenau_retry1" for i in range(3)}
                ),
                "four task average",
                four_task_average,
            ),
            Comparison(
                "short ViT diagnostics",
                "MAE BRIDGE minus base",
                load_named("rebuttal_compat_mae_vits_base"),
                load_named("rebuttal_compat_mae_vits_bridge"),
                "four task average",
                four_task_average,
            ),
        ]
    )

    for fraction, label in (("0.01", "1 percent labels"), ("0.10", "10 percent labels")):
        result.append(
            Comparison(
                "low label fine tuning",
                f"BRIDGE minus base at {label}",
                load_named(f"rebuttal_lowshot_base_{fraction}_gruenau_retry1"),
                load_named(f"rebuttal_lowshot_bridge_{fraction}_gruenau_retry1"),
                "ImageNet 100 LT accuracy",
                accuracy("imagenet100lt_test_accuracy"),
            )
        )

    ts_base = load_named("rebuttal_ts_cifar10lt_base_gruenau_retry1")
    ts_only = load_named("rebuttal_ts_cifar10lt_ts_gruenau_retry1")
    ts_bridge = load_named("rebuttal_ts_cifar10lt_bridge_gruenau_retry1")
    ts_combined = load_named("rebuttal_ts_cifar10lt_bridge_ts_gruenau_retry1")
    for metric_name, metric in (
        ("CIFAR 10 transfer", accuracy("cifar10r_test_accuracy")),
        ("CIFAR 100 transfer", accuracy("cifar100r_test_accuracy")),
    ):
        result.extend(
            [
                Comparison("TS factorial", "BRIDGE minus base", ts_base, ts_bridge, metric_name, metric),
                Comparison("TS factorial", "BRIDGE plus TS minus TS", ts_only, ts_combined, metric_name, metric),
                Comparison("TS factorial", "BRIDGE plus TS minus BRIDGE", ts_bridge, ts_combined, metric_name, metric),
            ]
        )

    generator_pairs = (
        (
            "FLUX dev 20 minus 6 steps",
            "rebuttal_flux_fair_dev_steps6_gruenau_retry1",
            "rebuttal_flux_fair_dev_steps20_gruenau_retry1",
        ),
        (
            "FLUX schnell guidance 3 minus guidance 1",
            "rebuttal_flux_fair_schnell_guidance1_gruenau_retry1",
            "rebuttal_flux_fair_schnell_guidance3_gruenau_retry1",
        ),
    )
    for label, control, treatment in generator_pairs:
        result.append(
            Comparison(
                "generator settings",
                label,
                load_named(control),
                load_named(treatment),
                "four task average",
                four_task_average,
            )
        )

    result.extend(
        [
            Comparison(
                "adaptive acquisition",
                "top tail minus adaptive mode window",
                load_named("rebuttal_feedback_adaptive_gruenau_retry1"),
                load_named("rebuttal_feedback_top_tail_gruenau_retry1"),
                "four task average",
                four_task_average,
            ),
            Comparison(
                "adaptive acquisition",
                "dense placebo minus adaptive mode window",
                load_named("rebuttal_feedback_adaptive_gruenau_retry1"),
                load_named("rebuttal_feedback_dense_placebo_gruenau_retry1"),
                "four task average",
                four_task_average,
            ),
            Comparison(
                "TADA controls",
                "BRIDGE plus TADA minus TADA",
                load_named("rebuttal_adaptive_tada_gruenau_retry1"),
                load_named("rebuttal_adaptive_bridge_tada_gruenau_retry1"),
                "four task average",
                four_task_average,
            ),
        ]
    )
    return result


def manuscript_summary_comparisons() -> list[SummaryComparison]:
    """Comparisons for which only reported mean and sample SD are retained."""

    result: list[SummaryComparison] = []

    main_averages = {
        "ImageNet 100 LT": ((39.8, 1.5), (42.5, 0.7)),
        "CIFAR 10 LT": ((29.1, 0.9), (34.2, 1.0)),
        "CIFAR 100 LT": ((23.8, 1.4), (34.4, 1.0)),
        "PASS 10k": ((36.8, 0.8), (39.9, 1.0)),
        "DiffusionDB 10k": ((37.7, 0.9), (40.1, 0.7)),
    }
    for source, (control, treatment) in main_averages.items():
        result.append(
            SummaryComparison(
                "main source averages",
                f"BRIDGE minus SimCLR on {source}",
                "seven task linear probe average",
                control[0],
                control[1],
                treatment[0],
                treatment[1],
            )
        )

    selector = {
        "q1 to 99": (42.1, 0.7),
        "q50 to 99": (42.2, 0.9),
        "q75 to 99": (42.3, 0.7),
        "top tail": (34.3, 1.2),
        "uniform": (35.5, 1.2),
    }
    for treatment_name in ("q1 to 99", "q50 to 99", "q75 to 99"):
        for control_name in ("top tail", "uniform"):
            control = selector[control_name]
            treatment = selector[treatment_name]
            result.append(
                SummaryComparison(
                    "selector summary",
                    f"{treatment_name} minus {control_name}",
                    "seven task linear probe average",
                    control[0],
                    control[1],
                    treatment[0],
                    treatment[1],
                )
            )
    for treatment_name in ("q1 to 99", "q50 to 99"):
        control = selector["q75 to 99"]
        treatment = selector[treatment_name]
        result.append(
            SummaryComparison(
                "selector summary",
                f"{treatment_name} minus q75 to 99",
                "seven task linear probe average",
                control[0],
                control[1],
                treatment[0],
                treatment[1],
            )
        )

    diagnostic = {
        "uniform SD3": (18.33, 0.32),
        "cluster frequency SD3": (18.66, 0.10),
        "mode window SD3": (18.88, 0.58),
        "oracle real restoration": (18.68, 1.74),
        "extreme score SD3": (19.51, 0.22),
        "TADA style SD3": (20.00, 0.98),
        "SD3 feature distillation": (18.02, 0.47),
        "distillation plus BRIDGE": (18.87, 0.45),
    }
    diagnostic_pairs = (
        ("uniform SD3", "mode window SD3"),
        ("oracle real restoration", "mode window SD3"),
        ("mode window SD3", "extreme score SD3"),
        ("mode window SD3", "TADA style SD3"),
        ("SD3 feature distillation", "mode window SD3"),
        ("mode window SD3", "distillation plus BRIDGE"),
    )
    for control_name, treatment_name in diagnostic_pairs:
        control = diagnostic[control_name]
        treatment = diagnostic[treatment_name]
        result.append(
            SummaryComparison(
                "one repair diagnostic summary",
                f"{treatment_name} minus {control_name}",
                "four task average",
                control[0],
                control[1],
                treatment[0],
                treatment[1],
            )
        )

    generators = {
        "FLUX dev 6 steps": (18.23, 0.98),
        "FLUX dev 20 steps": (18.78, 1.43),
        "FLUX schnell 20 steps": (18.85, 1.30),
        "FLUX schnell guidance 3": (19.68, 0.81),
        "SD3 6 steps": (18.58, 1.14),
        "SD3 20 steps": (19.70, 0.35),
    }
    generator_pairs = (
        ("FLUX dev 6 steps", "FLUX dev 20 steps"),
        ("FLUX schnell 20 steps", "FLUX schnell guidance 3"),
        ("FLUX dev 20 steps", "SD3 20 steps"),
        ("FLUX schnell 20 steps", "SD3 20 steps"),
        ("FLUX schnell guidance 3", "SD3 20 steps"),
        ("SD3 6 steps", "SD3 20 steps"),
    )
    for control_name, treatment_name in generator_pairs:
        control = generators[control_name]
        treatment = generators[treatment_name]
        result.append(
            SummaryComparison(
                "generator summary",
                f"{treatment_name} minus {control_name}",
                "four task average",
                control[0],
                control[1],
                treatment[0],
                treatment[1],
            )
        )

    return result


def main_table_cell_comparisons() -> list[SummaryComparison]:
    metric_names = (
        "CIFAR 10",
        "CIFAR 100",
        "Cars",
        "Aircraft",
        "Flowers",
        "Pets",
        "ImageNet 100 LT",
        "seven task average",
    )
    value_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*\{\\scriptsize\$\\pm\$\s*(\d+(?:\.\d+)?)\}"
    )
    result: list[SummaryComparison] = []
    for relative_path in (
        "paper_work/neurips_bridge_paper/tables/main_sota_label_sources.tex",
        "paper_work/neurips_bridge_paper/tables/main_sota_web_sources.tex",
    ):
        source = ""
        simclr: list[tuple[float, float]] | None = None
        for line in (REPO_ROOT / relative_path).read_text().splitlines():
            columns = line.split("&")
            if len(columns) < 2:
                continue
            first_column = columns[0].strip()
            method_column = columns[1]
            if first_column:
                source = first_column
            values = [(float(mean), float(sd)) for mean, sd in value_pattern.findall(line)]
            if "SimCLR" in method_column and len(values) == len(metric_names):
                simclr = values
            elif "BRIDGE (ours)" in method_column and len(values) == len(metric_names):
                if simclr is None:
                    raise RuntimeError(f"Missing SimCLR row before BRIDGE row for {source}")
                for metric_name, control, treatment in zip(metric_names, simclr, values):
                    result.append(
                        SummaryComparison(
                            f"main table cells {source}",
                            f"BRIDGE minus SimCLR on {source}",
                            metric_name,
                            control[0],
                            control[1],
                            treatment[0],
                            treatment[1],
                        )
                    )
                simclr = None
    return result


def format_number(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def create_report(rows: list[dict]) -> str:
    lines = [
        "# Statistical audit of rebuttal experiments",
        "",
        "The experimental unit is the independent training seed. Tests are two sided paired t tests because the compared runs share seeds and branch checkpoints. The table reports unadjusted p values and Holm adjusted p values within each experimental family. Confidence intervals use the t distribution. With only three seeds the intervals are necessarily wide and the exact Wilcoxon test has very low resolution.",
        "",
        "| Family | Comparison | Metric | n | Mean difference | 95% CI | p | Holm p | Cohen dz | Wilcoxon p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = "NA"
        if "ci95_low" in row:
            ci = f"[{format_number(row['ci95_low'])}, {format_number(row['ci95_high'])}]"
        lines.append(
            "| {family} | {comparison} | {metric} | {n} | {difference} | {ci} | {p} | {holm} | {dz} | {wilcoxon} |".format(
                family=row["family"],
                comparison=row["comparison"],
                metric=row["metric"],
                n=row["n"],
                difference=format_number(row.get("difference_mean", math.nan)),
                ci=ci,
                p=format_number(row.get("paired_t_p_two_sided", math.nan), 4),
                holm=format_number(row.get("holm_p", math.nan), 4),
                dz=format_number(row.get("cohen_dz", math.nan)),
                wilcoxon=format_number(row.get("wilcoxon_p_two_sided", math.nan), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "A small p value is not evidence that every downstream dataset improves. The primary four task average and each downstream task are reported separately. Holm adjustment is applied only within the named family and is not a substitute for identifying a primary endpoint in advance.",
            "",
            "Comparisons absent from this report either have fewer than two paired seeds in the retained result store, have only manuscript level mean and standard deviation summaries, or do not share an auditable protocol. They must not be described as statistically significant without recovering the raw paired seeds.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_summary_report(rows: list[dict]) -> str:
    lines = [
        "# Summary only statistical sensitivity analysis",
        "",
        "Raw paired seed values were not retained for these comparisons. The tests below therefore use two sided Welch tests reconstructed from the reported means, sample standard deviations, and three runs per condition. This ignores the pairing and should not replace a paired analysis if the original seeds can be recovered. It is included to show what the published summaries alone support.",
        "",
        "| Family | Comparison | Metric | Difference | 95% CI | Welch p | Holm p | Cohen d |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {comparison} | {metric} | {difference} | [{low}, {high}] | {p} | {holm} | {effect} |".format(
                family=row["family"],
                comparison=row["comparison"],
                metric=row["metric"],
                difference=format_number(row["difference_mean"]),
                low=format_number(row["ci95_low"]),
                high=format_number(row["ci95_high"]),
                p=format_number(row["welch_p_two_sided"], 4),
                holm=format_number(row.get("holm_p", math.nan), 4),
                effect=format_number(row["cohen_d"]),
            )
        )
    lines.extend(
        [
            "",
            "These p values are sensitivity checks rather than substitutes for the missing paired seed records. In particular, no one repair diagnostic or generator comparison should be called significant from these summaries after familywise correction.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [paired_result(item) for item in comparisons()]
    families = sorted({row["family"] for row in rows})
    for family in families:
        positions = [index for index, row in enumerate(rows) if row["family"] == family]
        adjusted = holm_adjust(
            [rows[index].get("paired_t_p_two_sided", math.nan) for index in positions]
        )
        for index, value in zip(positions, adjusted):
            rows[index]["holm_p"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "statistical_audit.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True) + "\n"
    )
    (args.output_dir / "statistical_audit.md").write_text(create_report(rows))

    summary_rows = [summary_result(item) for item in manuscript_summary_comparisons()]
    summary_families = sorted({row["family"] for row in summary_rows})
    for family in summary_families:
        positions = [index for index, row in enumerate(summary_rows) if row["family"] == family]
        adjusted = holm_adjust(
            [summary_rows[index].get("welch_p_two_sided", math.nan) for index in positions]
        )
        for index, value in zip(positions, adjusted):
            summary_rows[index]["holm_p"] = value
    (args.output_dir / "summary_only_audit.json").write_text(
        json.dumps(summary_rows, indent=2, allow_nan=True) + "\n"
    )
    (args.output_dir / "summary_only_audit.md").write_text(
        create_summary_report(summary_rows)
    )

    main_cell_rows = [summary_result(item) for item in main_table_cell_comparisons()]
    main_cell_families = sorted({row["family"] for row in main_cell_rows})
    for family in main_cell_families:
        positions = [index for index, row in enumerate(main_cell_rows) if row["family"] == family]
        adjusted = holm_adjust(
            [main_cell_rows[index].get("welch_p_two_sided", math.nan) for index in positions]
        )
        for index, value in zip(positions, adjusted):
            main_cell_rows[index]["holm_p"] = value
    (args.output_dir / "main_table_all_cells.json").write_text(
        json.dumps(main_cell_rows, indent=2, allow_nan=True) + "\n"
    )
    (args.output_dir / "main_table_all_cells.md").write_text(
        create_summary_report(main_cell_rows)
    )


if __name__ == "__main__":
    main()
