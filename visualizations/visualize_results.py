import matplotlib.pyplot as plt

# First plot data (Balanced, Imbalanced, New Method)
# ResNet50 data
simclr_data_main = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [42.47, 41.53, 51.83],
        "ViTBase": [39.85, 30.54, 44.37],
        "ResNet18": [45.15, 45.56, 45.09],
        "ResNet50": [57.45, 36.84, 53.11],  # Added ResNet50 data
    },
    "CIFAR10 KNN": {
        "ViTSmall": [29.95, 29.97, 40.29],
        "ViTBase": [28.55, 22.01, 31.97],
        "ResNet18": [36.90, 37.19, 35.92],
        "ResNet50": [46.36, 31.12, 43.94],  # Added ResNet50 data
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [15.79, 15.19, 25.08],
        "ViTBase": [11.26, 6.65, 18.08],
        "ResNet18": [17.69, 17.98, 18.38],
        "ResNet50": [26.74, 11.55, 21.33],  # Added ResNet50 data
    },
    "CIFAR100 KNN": {
        "ViTSmall": [8.26, 7.92, 13.45],
        "ViTBase": [5.66, 3.96, 7.76],
        "ResNet18": [11.72, 12.04, 11.52],
        "ResNet50": [17.08, 9.58, 14.81],  # Added ResNet50 data
    },
}

simclr_std_main = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [4.44, 2.28, 0.21],
        "ViTBase": [3.37, 6.73, 5.45],
        "ResNet18": [2.91, 1.38, 0.97],
        "ResNet50": [0.96, 10.29, 1.66],  # Added ResNet50 std
    },
    "CIFAR10 KNN": {
        "ViTSmall": [2.79, 1.60, 0.69],
        "ViTBase": [2.75, 5.25, 4.87],
        "ResNet18": [1.70, 1.89, 1.08],
        "ResNet50": [0.43, 15.30, 0.96],  # Added ResNet50 std
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [2.55, 0.86, 0.74],
        "ViTBase": [3.92, 0.99, 5.02],
        "ResNet18": [1.40, 0.93, 0.49],
        "ResNet50": [0.73, 5.56, 1.90],  # Added ResNet50 std
    },
    "CIFAR100 KNN": {
        "ViTSmall": [0.20, 0.43, 0.61],
        "ViTBase": [1.38, 1.62, 2.19],
        "ResNet18": [0.95, 0.45, 0.74],
        "ResNet50": [0.55, 7.87, 0.31],  # Added ResNet50 std
    },
}

simclr_data_ablation = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [51.83, 50.52, 52.25, 48.31],
        "ViTBase": [44.37, 22.52, 39.94, 38.56],
        "ResNet18": [45.09, 45.60, 48.09, 48.85],
        "ResNet50": [53.11, 51.06, 50.58, 43.89],  # Added ResNet50 data
    },
    "CIFAR10 KNN": {
        "ViTSmall": [40.29, 38.35, 38.82, 35.64],
        "ViTBase": [31.97, 18.02, 29.64, 27.89],
        "ResNet18": [35.92, 37.37, 40.10, 40.87],
        "ResNet50": [43.94, 39.75, 39.18, 33.56],  # Added ResNet50 data
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [25.08, 23.23, 24.59, 21.15],
        "ViTBase": [18.08, 4.22, 15.07, 14.91],
        "ResNet18": [18.38, 18.65, 20.16, 20.85],
        "ResNet50": [21.33, 21.59, 20.13, 16.50],  # Added ResNet50 data
    },
    "CIFAR100 KNN": {
        "ViTSmall": [13.45, 11.57, 12.44, 9.10],
        "ViTBase": [7.76, 3.16, 6.70, 6.65],
        "ResNet18": [11.52, 12.24, 13.32, 13.63],
        "ResNet50": [14.81, 12.89, 12.90, 10.34],  # Added ResNet50 data
    },
}

simclr_std_ablation = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [0.21, 1.44, 1.08, 1.25],
        "ViTBase": [5.45, 3.29, 8.96, 8.81],
        "ResNet18": [0.97, 2.30, 0.92, 1.15],
        "ResNet50": [1.66, 1.79, 2.88, 4.56],  # Added ResNet50 std
    },
    "CIFAR10 KNN": {
        "ViTSmall": [0.69, 1.12, 1.73, 1.42],
        "ViTBase": [4.87, 0.71, 6.00, 5.83],
        "ResNet18": [1.08, 2.20, 0.68, 1.40],
        "ResNet50": [0.96, 2.11, 3.78, 4.57],  # Added ResNet50 std
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [0.74, 1.88, 0.16, 0.91],
        "ViTBase": [5.02, 1.85, 5.99, 6.17],
        "ResNet18": [0.49, 1.35, 0.97, 1.10],
        "ResNet50": [1.90, 1.03, 2.00, 1.28],  # Added ResNet50 std
    },
    "CIFAR100 KNN": {
        "ViTSmall": [0.61, 0.48, 1.33, 0.88],
        "ViTBase": [2.19, 0.49, 1.88, 2.29],
        "ResNet18": [0.74, 0.98, 0.08, 0.97],
        "ResNet50": [0.31, 2.33, 1.98, 3.75],  # Added ResNet50 std
    },
}

# Colors and patterns
vitsmall_color = "#2ca02c"
vitbase_color = "#d62728"
resnet_color = "#1f77b4"  # Added ResNet18 color
resnet50_color = "#9467bd"
patterns = ["", "//", "oo", "xx"]


def create_single_plot(ax, data, std_data, title, methods):
    bar_width = 0.3  # Adjusted for four models
    group_gap = 0.3

    n_methods = len(data["ViTSmall"])
    vitsmall_start = -(n_methods * bar_width + group_gap)
    vitbase_start = vitsmall_start + n_methods * bar_width + group_gap
    resnet_start = vitbase_start + n_methods * bar_width + group_gap
    resnet50_start = (
        resnet_start + n_methods * bar_width + group_gap
    )  # Added ResNet-50 position

    for i in range(n_methods):
        # ViTSmall bars
        vitsmall_pos = vitsmall_start + i * bar_width
        vitsmall_val = data["ViTSmall"][i]
        vitsmall_std = std_data["ViTSmall"][i]

        vbar_small = ax.bar(
            vitsmall_pos, vitsmall_val, bar_width, color=vitsmall_color, alpha=0.7
        )
        vbar_small[0].set_hatch(patterns[i])

        ax.errorbar(
            vitsmall_pos,
            vitsmall_val,
            yerr=vitsmall_std,
            color="black",
            capsize=5,
            fmt="none",
            capthick=1,
            zorder=2,
        )
        ax.text(
            vitsmall_pos,
            vitsmall_val + vitsmall_std + 0.5,
            f"{vitsmall_val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

        # ViTBase bars
        vitbase_pos = vitbase_start + i * bar_width
        vitbase_val = data["ViTBase"][i]
        vitbase_std = std_data["ViTBase"][i]

        vbar_base = ax.bar(
            vitbase_pos, vitbase_val, bar_width, color=vitbase_color, alpha=0.7
        )
        vbar_base[0].set_hatch(patterns[i])

        ax.errorbar(
            vitbase_pos,
            vitbase_val,
            yerr=vitbase_std,
            color="black",
            capsize=5,
            fmt="none",
            capthick=1,
            zorder=2,
        )
        ax.text(
            vitbase_pos,
            vitbase_val + vitbase_std + 0.5,
            f"{vitbase_val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

        # ResNet18 bars
        resnet_pos = resnet_start + i * bar_width
        resnet_val = data["ResNet18"][i]
        resnet_std = std_data["ResNet18"][i]

        rbar = ax.bar(resnet_pos, resnet_val, bar_width, color=resnet_color, alpha=0.7)
        rbar[0].set_hatch(patterns[i])

        ax.errorbar(
            resnet_pos,
            resnet_val,
            yerr=resnet_std,
            color="black",
            capsize=5,
            fmt="none",
            capthick=1,
            zorder=2,
        )
        ax.text(
            resnet_pos,
            resnet_val + resnet_std + 0.5,
            f"{resnet_val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

        # ResNet-50 bars
        resnet50_pos = resnet50_start + i * bar_width
        resnet50_val = data["ResNet50"][i]
        resnet50_std = std_data["ResNet50"][i]

        rbar50 = ax.bar(
            resnet50_pos, resnet50_val, bar_width, color=resnet50_color, alpha=0.7
        )
        rbar50[0].set_hatch(patterns[i])

        ax.errorbar(
            resnet50_pos,
            resnet50_val,
            yerr=resnet50_std,
            color="black",
            capsize=5,
            fmt="none",
            capthick=1,
            zorder=2,
        )
        ax.text(
            resnet50_pos,
            resnet50_val + resnet50_std + 0.5,
            f"{resnet50_val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, pad=20)

    # Set x-ticks at the center of each group
    vitsmall_center = vitsmall_start + (n_methods * bar_width) / 2 - bar_width / 2
    vitbase_center = vitbase_start + (n_methods * bar_width) / 2 - bar_width / 2
    resnet_center = resnet_start + (n_methods * bar_width) / 2 - bar_width / 2
    resnet50_center = resnet50_start + (n_methods * bar_width) / 2 - bar_width / 2
    ax.set_xticks([vitsmall_center, vitbase_center, resnet_center, resnet50_center])
    ax.set_xticklabels(["ViT-Small", "ViT-Base", "ResNet18", "ResNet50"])


def create_plot(data, std_data, methods, title_suffix):
    # Increase the overall font size for all text in the plots
    plt.rcParams.update(
        {
            "font.size": 14,  # General font size
            "axes.titlesize": 14,  # Title font size
            "axes.labelsize": 14,  # Axis label font size
            "xtick.labelsize": 12,  # x-tick label font size
            "ytick.labelsize": 12,  # y-tick label font size
            "legend.fontsize": 14,  # Legend font size
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(16, 14))

    for i, (title, plot_data) in enumerate(data.items()):
        row = i // 2
        col = i % 2
        create_single_plot(axs[row, col], plot_data, std_data[title], title, methods)

    # Create legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#666666", alpha=0.6, hatch=p)
        for p in patterns
    ]
    fig.legend(
        handles,
        methods,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=4,
        markerscale=2.5,
    )

    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.2, hspace=0.5, wspace=0.3
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3, wspace=0.2)
    return fig


# Create main plot
main_methods = ["Balanced", "Imbalanced", "New Method"]
fig_main = create_plot(simclr_data_main, simclr_std_main, main_methods, "Main Methods")
plt.savefig("imagenet-100-main.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# Create ablation plot with New Method first
ablation_methods = ["New Method", "OOD Aug.", "Uniform Gen.", "Stepwise"]
fig_ablation = create_plot(
    simclr_data_ablation, simclr_std_ablation, ablation_methods, "Ablation Methods"
)
plt.savefig("imagenet-100-ablations.pdf", format="pdf", bbox_inches="tight")
# plt.show()
