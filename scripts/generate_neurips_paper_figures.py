#!/usr/bin/env python3
"""Generate paper-specific figures for the NeurIPS BRIDGE draft."""

import csv
import os
import pickle
import sys
import zipfile
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.gridspec import GridSpec

RESULTS_CSV = "paper_work/results_export/paper_results_metrics.csv"
OUTPUT_DIR = "paper_work/neurips_bridge_paper/figures"
HISTORY_PATH = (
    "visualizations/data/"
    "sota_imagenet-100-lt_simclr_clane9_imagenet-100_2026-02-16_10-23-17.881959/"
    "run_2/ood_distance_distribution/history.pt"
)
CYCLE_HISTORY_PATH = (
    "visualizations/data/"
    "sota_imagenet-100-lt_ts_clane9_imagenet-100_2026-02-16_10-31-59.405111/"
    "run_2/ood_distance_distribution/history.pt"
)

LINEAR_ALL = [
    "cifar10r_test_accuracy",
    "cifar100r_test_accuracy",
    "cars_test_accuracy",
    "aircraft_test_accuracy",
    "flowers_test_accuracy",
    "pets_test_accuracy",
    "imagenet100lt_test_accuracy",
]

KNN_ALL = [
    "cifar10knn_knn_test_accuracy",
    "cifar100knn_knn_test_accuracy",
    "carsknn_knn_test_accuracy",
    "aircraftknn_knn_test_accuracy",
    "flowersknn_knn_test_accuracy",
    "petsknn_knn_test_accuracy",
    "imagenet100ltknn_knn_test_accuracy",
]

SOTA_GROUPS = OrderedDict(
    [
        ("ImageNet-100-LT", "sota/imagenet-100-lt"),
        ("CIFAR-10-LT", "sota/cifar-10-lt"),
        ("CIFAR-100-LT", "sota/cifar-100-lt"),
        ("PASS-10k", "sota/pass-subset"),
        ("DiffusionDB-10k", "sota/diffusiondb-subset"),
    ]
)

ABLATION_GROUPS = OrderedDict(
    [
        (
            "SSL objective",
            OrderedDict(
                [
                    ("ablations/pretraining/simclr", "SimCLR"),
                    ("ablations/pretraining/moco", "MoCo"),
                    ("ablations/pretraining/dino", "DINO"),
                ]
            ),
        ),
        (
            "Generation",
            OrderedDict(
                [
                    ("ablations/generation/stable_diffusion_3", "SD3"),
                    ("ablations/generation/flux", "FLUX"),
                    ("ablations/generation/repopulation", "Re-pop."),
                ]
            ),
        ),
        (
            "Selection",
            OrderedDict(
                [
                    ("ablations/sample_selection/mode_window", "Mode"),
                    ("ablations/sample_selection/ood_top", "Top tail"),
                    ("ablations/sample_selection/uniform", "Uniform"),
                ]
            ),
        ),
    ]
)


matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_results():
    by_exp = defaultdict(dict)
    with open(RESULTS_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            by_exp[row["experiment"]][row["metric"]] = (
                float(row["mean"]),
                float(row["std"]),
            )
    return by_exp


def average_metric(exp_metrics, metric_names):
    return float(sum(exp_metrics[name][0] for name in metric_names) / len(metric_names))


def average_std(exp_metrics, metric_names):
    return float(sum(exp_metrics[name][1] for name in metric_names) / len(metric_names))


def load_history(path):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    with zipfile.ZipFile(path) as archive:
        history = pickle.loads(archive.read("history/data.pkl"))
    return history


def mode_window_selection(distances, num_samples=500, quantile_range=(0.01, 0.99)):
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        raise ValueError("No valid distances available for selector figure")

    sorted_distances = np.sort(distances)
    clipped_low, clipped_high = np.quantile(sorted_distances, list(quantile_range))
    histogram_values = sorted_distances[
        (sorted_distances >= clipped_low) & (sorted_distances <= clipped_high)
    ]
    if histogram_values.size < 2 or np.isclose(clipped_low, clipped_high):
        histogram_values = sorted_distances
        clipped_low = float(sorted_distances[0])
        clipped_high = float(sorted_distances[-1])

    candidate_edges = np.histogram_bin_edges(histogram_values, bins="auto")
    bins = max(8, min(512, len(candidate_edges) - 1, histogram_values.size))
    hist_counts, bin_edges = np.histogram(
        histogram_values,
        bins=bins,
        range=(clipped_low, clipped_high),
        density=False,
    )

    mode_bin_idx = int(np.argmax(hist_counts))
    mode_left = float(bin_edges[mode_bin_idx])
    mode_right = float(bin_edges[mode_bin_idx + 1])
    mode_center = 0.5 * (mode_left + mode_right)

    selected_indices = np.argsort(np.abs(sorted_distances - mode_center))[:num_samples]
    mode_selected = sorted_distances[np.sort(selected_indices)]
    top_tail = np.sort(sorted_distances)[-num_samples:]

    return {
        "distances": sorted_distances,
        "histogram_values": histogram_values,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
        "mode_center": mode_center,
        "mode_selected": mode_selected,
        "top_tail": top_tail,
    }


def save_figure(fig, stem):
    png_path = os.path.join(OUTPUT_DIR, stem + ".png")
    pdf_path = os.path.join(OUTPUT_DIR, stem + ".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor="#2f3e46"):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#14213d",
        wrap=True,
    )


def draw_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#2f3e46", linewidth=1.5),
    )


def plot_method_overview():
    history = load_history(HISTORY_PATH)
    selector = mode_window_selection(history[-1]["distances"])

    fig = plt.figure(figsize=(12.5, 4.5))
    grid = GridSpec(1, 2, width_ratios=[1.15, 1.1], wspace=0.22, figure=fig)

    ax0 = fig.add_subplot(grid[0, 0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")

    draw_rounded_box(
        ax0, 0.01, 0.66, 0.25, 0.2,
        "1. Pretrain encoder\non current source set",
        "#d9f0ff",
    )
    draw_rounded_box(
        ax0, 0.31, 0.66, 0.25, 0.2,
        "2. Embed data and\ncompute mean\nkNN distance",
        "#fef3c7",
    )
    draw_rounded_box(
        ax0, 0.61, 0.66, 0.28, 0.2,
        "3. Build clipped histogram\nand locate the modal\nsparse region",
        "#fde2e4",
    )
    draw_rounded_box(
        ax0, 0.18, 0.20, 0.30, 0.2,
        "4. Select 500 samples\nclosest to the modal bin",
        "#d8f3dc",
    )
    draw_rounded_box(
        ax0, 0.56, 0.20, 0.31, 0.2,
        "5. Generate 5 variants per\nselected image with frozen SD3",
        "#ede9fe",
    )
    ax0.text(
        0.50,
        0.03,
        "Repeat for 5 cycles and evaluate the final encoder by linear probe and kNN transfer.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#334155",
    )
    draw_arrow(ax0, (0.26, 0.76), (0.31, 0.76))
    draw_arrow(ax0, (0.56, 0.76), (0.61, 0.76))
    draw_arrow(ax0, (0.78, 0.66), (0.66, 0.41))
    draw_arrow(ax0, (0.45, 0.41), (0.33, 0.40))
    draw_arrow(ax0, (0.48, 0.30), (0.56, 0.30))
    draw_arrow(ax0, (0.87, 0.30), (0.94, 0.56))
    ax0.text(0.02, 0.94, "BRIDGE training loop", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(grid[0, 1])
    distances = selector["distances"]
    plot_low = float(np.quantile(distances, 0.001))
    plot_high = float(np.quantile(distances, 0.995))
    bins = np.linspace(plot_low, plot_high, 55)
    ax1.hist(
        distances[(distances >= plot_low) & (distances <= plot_high)],
        bins=bins,
        color="#d8dee9",
        edgecolor="#475569",
        linewidth=0.7,
        alpha=0.95,
    )
    ax1.axvspan(
        float(selector["mode_selected"].min()),
        float(selector["mode_selected"].max()),
        color="#2a9d8f",
        alpha=0.24,
        label="Mode window",
    )
    ax1.axvspan(
        float(selector["top_tail"].min()),
        plot_high,
        color="#e76f51",
        alpha=0.18,
        label="Top tail",
    )
    ax1.axvline(
        selector["mode_center"],
        color="#1d3557",
        linestyle="--",
        linewidth=1.3,
        label="Histogram mode",
    )
    ax1.set_xlabel("Mean kNN distance")
    ax1.set_ylabel("Count")
    ax1.set_title("Mode window selection avoids the extreme tail")
    ax1.set_xlim(plot_low, plot_high)
    ax1.legend(frameon=False, loc="upper right")

    save_figure(fig, "method_overview")


def plot_source_regime_gain(by_exp):
    labels = []
    deltas = []
    for label, prefix in SOTA_GROUPS.items():
        bridge_exp = "%s/bridge" % prefix
        simclr_exp = "%s/simclr" % prefix
        bridge_avg = average_metric(by_exp[bridge_exp], LINEAR_ALL)
        simclr_avg = average_metric(by_exp[simclr_exp], LINEAR_ALL)
        labels.append(label.replace("-10k", ""))
        deltas.append(100.0 * (bridge_avg - simclr_avg))

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color="#1d4ed8", alpha=0.88, width=0.62)
    ax.axhline(0.0, color="#64748b", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average linear gain over SimCLR")
    ax.set_title("BRIDGE improves average transfer in every source regime")
    for bar, value in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.12,
            "%+.2f" % value,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, "source_regime_gain")


def plot_ablation_tradeoffs(by_exp):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=False)
    linear_color = "#2563eb"
    knn_color = "#94a3b8"

    for ax, (title, group) in zip(axes, ABLATION_GROUPS.items()):
        labels = []
        linear_vals = []
        linear_err = []
        knn_vals = []
        knn_err = []
        for exp, label in group.items():
            labels.append(label)
            linear_vals.append(100.0 * average_metric(by_exp[exp], LINEAR_ALL))
            linear_err.append(100.0 * average_std(by_exp[exp], LINEAR_ALL))
            knn_vals.append(100.0 * average_metric(by_exp[exp], KNN_ALL))
            knn_err.append(100.0 * average_std(by_exp[exp], KNN_ALL))

        y = np.arange(len(labels))
        ax.barh(y + 0.18, linear_vals, height=0.32, color=linear_color, label="Avg. linear")
        ax.barh(y - 0.18, knn_vals, height=0.32, color=knn_color, label="Avg. kNN")
        ax.errorbar(linear_vals, y + 0.18, xerr=linear_err, fmt="none", ecolor="#0f172a", capsize=2)
        ax.errorbar(knn_vals, y - 0.18, xerr=knn_err, fmt="none", ecolor="#475569", capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(0, max(linear_vals + knn_vals) + 8.0)
        ax.grid(axis="x", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Setting")
    axes[1].set_xlabel("Average transfer score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "ablation_tradeoffs")


def plot_cycle_histories():
    history = load_history(CYCLE_HISTORY_PATH)
    fig, axes = plt.subplots(1, len(history), figsize=(11.2, 3.2), sharey=True)

    combined = []
    for item in history:
        d = np.asarray(item["distances"], dtype=np.float64)
        d = d[np.isfinite(d)]
        combined.append(d)
    plot_low = min(float(np.quantile(d, 0.01)) for d in combined)
    plot_high = max(float(np.quantile(d, 0.99)) for d in combined)
    bins = np.linspace(plot_low, plot_high, 42)

    for ax, item, distances in zip(axes, history, combined):
        ax.hist(distances, bins=bins, density=True, color="#cbd5e1", edgecolor="#475569", linewidth=0.7)
        ax.axvline(float(np.mean(distances)), color="#1d4ed8", linewidth=1.4)
        ax.axvline(float(np.quantile(distances, 0.99)), color="#e76f51", linestyle="--", linewidth=1.2)
        ax.set_title("Cycle %d" % int(item["cycle"]))
        ax.set_xlabel("Mean kNN distance")
    axes[0].set_ylabel("Density")
    fig.suptitle("Representative OOD histograms shift across cycles")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "cycle_histories_example")


def main():
    ensure_output_dir()
    by_exp = load_results()
    plot_method_overview()
    plot_source_regime_gain(by_exp)
    plot_ablation_tradeoffs(by_exp)
    plot_cycle_histories()


if __name__ == "__main__":
    main()
