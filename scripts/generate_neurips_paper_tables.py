#!/usr/bin/env python3
"""Generate LaTeX tables for the NeurIPS BRIDGE paper."""

import csv
import os
from collections import OrderedDict, defaultdict


RESULTS_CSV = "paper_work/results_export/paper_results_metrics.csv"
OUTPUT_DIR = "paper_work/neurips_bridge_paper/tables"

LINEAR_ALL = [
    ("cifar10r_test_accuracy", "C10"),
    ("cifar100r_test_accuracy", "C100"),
    ("cars_test_accuracy", "Cars"),
    ("aircraft_test_accuracy", "Aircraft"),
    ("flowers_test_accuracy", "Flowers"),
    ("pets_test_accuracy", "Pets"),
    ("imagenet100lt_test_accuracy", "IN100-LT"),
]

KNN_ALL = [
    ("cifar10knn_knn_test_accuracy", "C10"),
    ("cifar100knn_knn_test_accuracy", "C100"),
    ("carsknn_knn_test_accuracy", "Cars"),
    ("aircraftknn_knn_test_accuracy", "Aircraft"),
    ("flowersknn_knn_test_accuracy", "Flowers"),
    ("petsknn_knn_test_accuracy", "Pets"),
    ("imagenet100ltknn_knn_test_accuracy", "IN100-LT"),
]

LINEAR_MAIN = [
    ("cifar10r_test_accuracy", "C10"),
    ("cifar100r_test_accuracy", "C100"),
    ("flowers_test_accuracy", "Flowers"),
    ("pets_test_accuracy", "Pets"),
    ("imagenet100lt_test_accuracy", "IN100-LT"),
]

KNN_MAIN = [
    ("cifar10knn_knn_test_accuracy", "C10"),
    ("cifar100knn_knn_test_accuracy", "C100"),
    ("flowersknn_knn_test_accuracy", "Flowers"),
    ("petsknn_knn_test_accuracy", "Pets"),
    ("imagenet100ltknn_knn_test_accuracy", "IN100-LT"),
]

BASELINE_ROWS = OrderedDict(
    [
        ("baseline/balanced", "Balanced"),
        ("baseline/imbalanced", "Imbalanced"),
        ("baseline/newmethod_imbalanced", "BRIDGE (ours)"),
    ]
)

SOTA_GROUPS = OrderedDict(
    [
        ("ImageNet-100-LT", "sota/imagenet-100-lt"),
        ("CIFAR-10-LT", "sota/cifar-10-lt"),
        ("CIFAR-100-LT", "sota/cifar-100-lt"),
        ("PASS-10k", "sota/pass-subset"),
        ("DiffusionDB-10k", "sota/diffusiondb-subset"),
    ]
)

SOTA_METHODS = OrderedDict(
    [
        ("simclr", "SimCLR"),
        ("ts", "TS"),
        ("sdclr", "SDCLR"),
        ("bridge", "BRIDGE (ours)"),
        ("bridge_ts", "BRIDGE+TS (ours)"),
        ("bridge_sdclr", "BRIDGE+SDCLR (ours)"),
    ]
)

ABLATION_GROUPS = OrderedDict(
    [
        (
            "pretraining",
            OrderedDict(
                [
                    ("ablations/pretraining/simclr", "SimCLR"),
                    ("ablations/pretraining/moco", "MoCo"),
                    ("ablations/pretraining/dino", "DINO"),
                ]
            ),
        ),
        (
            "generation",
            OrderedDict(
                [
                    ("ablations/generation/stable_diffusion_3", "Stable Diffusion 3"),
                    ("ablations/generation/flux", "FLUX"),
                    ("ablations/generation/repopulation", "Re-population"),
                ]
            ),
        ),
        (
            "selection",
            OrderedDict(
                [
                    ("ablations/sample_selection/mode_window", "Mode window"),
                    ("ablations/sample_selection/ood_top", "Top tail"),
                    ("ablations/sample_selection/uniform", "Uniform"),
                ]
            ),
        ),
        (
            "cycles",
            OrderedDict(
                [
                    ("ablations/cycles/cycles_2", "2 cycles"),
                    ("ablations/cycles/cycles_10", "10 cycles"),
                    ("ablations/cycles/cycles_20", "20 cycles"),
                    ("baseline/newmethod_imbalanced", "5 cycles"),
                ]
            ),
        ),
        (
            "architecture",
            OrderedDict(
                [
                    ("ablations/architecture/resnet18", "ResNet-18"),
                    ("ablations/architecture/resnet50", "ResNet-50"),
                    ("ablations/architecture/vit_s", "ViT-S"),
                    ("ablations/architecture/vit_b", "ViT-B"),
                ]
            ),
        ),
    ]
)


def load_results():
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    by_exp = defaultdict(dict)
    for row in rows:
        by_exp[row["experiment"]][row["metric"]] = (
            float(row["mean"]),
            float(row["std"]),
            row["complete"] == "True",
        )
    return by_exp


def fmt(mean, std, bold=False):
    cell = "%.1f {\\scriptsize$\\pm$ %.1f}" % (100.0 * mean, 100.0 * std)
    if bold:
        return "\\textbf{%s}" % cell
    return cell


def avg_for_metrics(exp_metrics, metric_names):
    vals = [exp_metrics[m][0] for m in metric_names]
    return sum(vals) / float(len(vals))


def avg_std_for_metrics(exp_metrics, metric_names):
    vals = [exp_metrics[m][1] for m in metric_names]
    return sum(vals) / float(len(vals))


def default_average_metric_names(metrics):
    metric_names = [m for m, _ in metrics]
    knn_names = [m for m, _ in KNN_ALL]
    if all(name in knn_names for name in metric_names):
        return knn_names
    return [m for m, _ in LINEAR_ALL]


def build_column_spec(ncols, leading="l"):
    return leading + " " + " ".join(["c"] * (ncols - 1))


def write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def render_table(caption, label, headers, body_lines, colspec, position="t", size="\\scriptsize"):
    lines = [
        "\\begin{table}[%s]" % position,
        "\\centering",
        size,
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{%s}" % caption,
        "\\label{%s}" % label,
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{%s}" % colspec,
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(body_lines)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def baseline_table(by_exp, metrics, filename, caption, label):
    headers = ["Method"] + [name for _, name in metrics] + ["Avg."]
    metric_names = default_average_metric_names(metrics)
    body = []
    best_by_metric = {}
    for metric, _ in metrics:
        best_by_metric[metric] = max(by_exp[exp][metric][0] for exp in BASELINE_ROWS)
    best_avg = max(avg_for_metrics(by_exp[exp], metric_names) for exp in BASELINE_ROWS)

    for exp, display in BASELINE_ROWS.items():
        row = [display]
        for metric, _ in metrics:
            mean, std, _ = by_exp[exp][metric]
            row.append(fmt(mean, std, abs(mean - best_by_metric[metric]) < 1e-9))
        avg_mean = avg_for_metrics(by_exp[exp], metric_names)
        avg_std = avg_std_for_metrics(by_exp[exp], metric_names)
        row.append(fmt(avg_mean, avg_std, abs(avg_mean - best_avg) < 1e-9))
        body.append(" & ".join(row) + " \\\\")

    text = render_table(caption, label, headers, body, build_column_spec(len(headers)))
    write(os.path.join(OUTPUT_DIR, filename), text)


def sota_table(by_exp, metrics, filename, caption, label):
    headers = ["Source", "Method"] + [name for _, name in metrics] + ["Avg."]
    body = []
    metric_names = default_average_metric_names(metrics)

    for group_name, prefix in SOTA_GROUPS.items():
        group_exps = ["%s/%s" % (prefix, key) for key in SOTA_METHODS]
        best_by_metric = {}
        for metric, _ in metrics:
            best_by_metric[metric] = max(by_exp[exp][metric][0] for exp in group_exps)
        best_avg = max(avg_for_metrics(by_exp[exp], metric_names) for exp in group_exps)

        for idx, (method_key, method_name) in enumerate(SOTA_METHODS.items()):
            exp = "%s/%s" % (prefix, method_key)
            row = [group_name if idx == 0 else "", method_name]
            for metric, _ in metrics:
                mean, std, _ = by_exp[exp][metric]
                row.append(fmt(mean, std, abs(mean - best_by_metric[metric]) < 1e-9))
            avg_mean = avg_for_metrics(by_exp[exp], metric_names)
            avg_std = avg_std_for_metrics(by_exp[exp], metric_names)
            row.append(fmt(avg_mean, avg_std, abs(avg_mean - best_avg) < 1e-9))
            body.append(" & ".join(row) + " \\\\")
        if group_name != list(SOTA_GROUPS.keys())[-1]:
            body.append("\\midrule")

    text = render_table(
        caption,
        label,
        headers,
        body,
        build_column_spec(len(headers), leading="l l"),
        position="t",
        size="\\tiny",
    )
    write(os.path.join(OUTPUT_DIR, filename), text)


def sota_average_table(by_exp):
    headers = ["Source", "SimCLR", "TS", "SDCLR", "BRIDGE", "BRIDGE+TS", "BRIDGE+SDCLR"]
    body = []
    metric_names = [m for m, _ in LINEAR_ALL]

    for group_name, prefix in SOTA_GROUPS.items():
        group_exps = ["%s/%s" % (prefix, key) for key in SOTA_METHODS]
        best_avg = max(avg_for_metrics(by_exp[exp], metric_names) for exp in group_exps)
        row = [group_name]
        for method_key in SOTA_METHODS:
            exp = "%s/%s" % (prefix, method_key)
            avg_mean = avg_for_metrics(by_exp[exp], metric_names)
            avg_std = avg_std_for_metrics(by_exp[exp], metric_names)
            row.append(fmt(avg_mean, avg_std, abs(avg_mean - best_avg) < 1e-9))
        body.append(" & ".join(row) + " \\\\")

    text = render_table(
        "Average linear-probe transfer over all seven downstream tasks. The full per-task linear and $k$NN tables are deferred to the appendix.",
        "tab:main_sota_average",
        headers,
        body,
        "l c c c c c c",
        position="t",
        size="\\scriptsize",
    )
    write(os.path.join(OUTPUT_DIR, "main_sota_average.tex"), text)


def ablation_table(by_exp, group_key, metrics, filename, caption, label):
    group = ABLATION_GROUPS[group_key]
    headers = ["Setting"] + [name for _, name in metrics] + ["Avg."]
    body = []
    metric_names = default_average_metric_names(metrics)
    best_by_metric = {}
    for metric, _ in metrics:
        best_by_metric[metric] = max(by_exp[exp][metric][0] for exp in group)
    best_avg = max(avg_for_metrics(by_exp[exp], metric_names) for exp in group)

    for exp, display in group.items():
        row = [display]
        for metric, _ in metrics:
            mean, std, _ = by_exp[exp][metric]
            row.append(fmt(mean, std, abs(mean - best_by_metric[metric]) < 1e-9))
        avg_mean = avg_for_metrics(by_exp[exp], metric_names)
        avg_std = avg_std_for_metrics(by_exp[exp], metric_names)
        row.append(fmt(avg_mean, avg_std, abs(avg_mean - best_avg) < 1e-9))
        body.append(" & ".join(row) + " \\\\")

    text = render_table(caption, label, headers, body, build_column_spec(len(headers)))
    write(os.path.join(OUTPUT_DIR, filename), text)


def main_ablation_summary(by_exp):
    headers = ["Group", "Setting", "Avg. Lin.", "Avg. kNN", "Flowers", "Pets", "IN100-LT"]
    linear_metric_names = [m for m, _ in LINEAR_ALL]
    knn_metric_names = [m for m, _ in KNN_ALL]
    body = []

    summary_groups = ["pretraining", "generation", "selection"]
    for group_key in summary_groups:
        group = ABLATION_GROUPS[group_key]
        best_lin = max(avg_for_metrics(by_exp[exp], linear_metric_names) for exp in group)
        best_knn = max(avg_for_metrics(by_exp[exp], knn_metric_names) for exp in group)
        best_flowers = max(by_exp[exp]["flowers_test_accuracy"][0] for exp in group)
        best_pets = max(by_exp[exp]["pets_test_accuracy"][0] for exp in group)
        best_in100 = max(by_exp[exp]["imagenet100lt_test_accuracy"][0] for exp in group)

        group_name = {
            "pretraining": "SSL objective",
            "generation": "Generation",
            "selection": "Selection",
        }[group_key]

        for idx, (exp, display) in enumerate(group.items()):
            avg_lin = avg_for_metrics(by_exp[exp], linear_metric_names)
            avg_knn = avg_for_metrics(by_exp[exp], knn_metric_names)
            flowers_mean, flowers_std, _ = by_exp[exp]["flowers_test_accuracy"]
            pets_mean, pets_std, _ = by_exp[exp]["pets_test_accuracy"]
            in_mean, in_std, _ = by_exp[exp]["imagenet100lt_test_accuracy"]
            row = [
                group_name if idx == 0 else "",
                display,
                fmt(avg_lin, avg_std_for_metrics(by_exp[exp], linear_metric_names), abs(avg_lin - best_lin) < 1e-9),
                fmt(avg_knn, avg_std_for_metrics(by_exp[exp], knn_metric_names), abs(avg_knn - best_knn) < 1e-9),
                fmt(flowers_mean, flowers_std, abs(flowers_mean - best_flowers) < 1e-9),
                fmt(pets_mean, pets_std, abs(pets_mean - best_pets) < 1e-9),
                fmt(in_mean, in_std, abs(in_mean - best_in100) < 1e-9),
            ]
            body.append(" & ".join(row) + " \\\\")
        if group_key != summary_groups[-1]:
            body.append("\\midrule")

    text = render_table(
        "Summary of the most important ablations on ImageNet-100-LT. Average columns are computed over all seven downstream tasks.",
        "tab:main_ablation_summary",
        headers,
        body,
        "l l c c c c c",
        position="t",
        size="\\scriptsize",
    )
    write(os.path.join(OUTPUT_DIR, "main_ablation_summary.tex"), text)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    by_exp = load_results()

    baseline_table(
        by_exp,
        LINEAR_MAIN,
        "main_baselines_linear.tex",
        "Linear probe comparison between balanced pretraining, imbalanced pretraining, and BRIDGE on ImageNet-100-LT. The reported average is over all seven downstream linear-probe tasks, including Cars and Aircraft.",
        "tab:main_baselines_linear",
    )
    baseline_table(
        by_exp,
        LINEAR_ALL,
        "app_baselines_linear_full.tex",
        "Full linear-probe baseline comparison on ImageNet-100-LT pretraining.",
        "tab:app_baselines_linear_full",
    )
    baseline_table(
        by_exp,
        KNN_ALL,
        "app_baselines_knn_full.tex",
        "Full kNN baseline comparison on ImageNet-100-LT pretraining.",
        "tab:app_baselines_knn_full",
    )

    sota_table(
        by_exp,
        LINEAR_MAIN,
        "main_sota_linear.tex",
        "Linear-probe transfer for all source datasets. PASS and DiffusionDB use 10k examples with a 100/0/0 pretraining split. The average is computed over all seven downstream linear-probe tasks.",
        "tab:main_sota_linear",
    )
    sota_average_table(by_exp)
    sota_table(
        by_exp,
        LINEAR_ALL,
        "app_sota_linear_full.tex",
        "Full linear-probe transfer results across all source datasets and comparison methods.",
        "tab:app_sota_linear_full",
    )
    sota_table(
        by_exp,
        KNN_ALL,
        "app_sota_knn_full.tex",
        "Full kNN transfer results across all source datasets and comparison methods.",
        "tab:app_sota_knn_full",
    )

    ablation_table(
        by_exp,
        "pretraining",
        LINEAR_MAIN,
        "main_ablation_pretraining.tex",
        "Effect of the SSL pretraining objective inside BRIDGE on ImageNet-100-LT.",
        "tab:main_ablation_pretraining",
    )
    ablation_table(
        by_exp,
        "generation",
        LINEAR_MAIN,
        "main_ablation_generation.tex",
        "Effect of the augmentation mechanism inside BRIDGE on ImageNet-100-LT.",
        "tab:main_ablation_generation",
    )
    ablation_table(
        by_exp,
        "selection",
        LINEAR_MAIN,
        "main_ablation_selection.tex",
        "Effect of sample-selection strategy inside BRIDGE on ImageNet-100-LT.",
        "tab:main_ablation_selection",
    )
    main_ablation_summary(by_exp)

    for group_key, group_name in [
        ("pretraining", "Pretraining objective"),
        ("generation", "Augmentation mechanism"),
        ("selection", "Sample-selection strategy"),
        ("cycles", "Cycle count"),
        ("architecture", "Backbone architecture"),
    ]:
        ablation_table(
            by_exp,
            group_key,
            LINEAR_ALL,
            "app_%s_linear.tex" % group_key,
            "%s ablation with full linear-probe results." % group_name,
            "tab:app_%s_linear" % group_key,
        )
        ablation_table(
            by_exp,
            group_key,
            KNN_ALL,
            "app_%s_knn.tex" % group_key,
            "%s ablation with full kNN results." % group_name,
            "tab:app_%s_knn" % group_key,
        )


if __name__ == "__main__":
    main()
