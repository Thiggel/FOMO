import matplotlib.pyplot as plt

# First plot data (Balanced, Imbalanced, New Method)
simclr_data_main = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [42.47, 41.53, 51.83],
        "ViTBase": [39.85, 30.54, 44.37],
        "ResNet18": [45.15, 45.56, 45.09],  # Added ResNet18 data
    },
    "CIFAR10 KNN": {
        "ViTSmall": [29.95, 29.97, 40.29],
        "ViTBase": [28.55, 22.01, 31.97],
        "ResNet18": [36.90, 37.19, 35.92],  # Added ResNet18 data
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [15.79, 15.19, 25.08],
        "ViTBase": [11.26, 6.65, 18.08],
        "ResNet18": [17.69, 17.98, 18.38],  # Added ResNet18 data
    },
    "CIFAR100 KNN": {
        "ViTSmall": [8.26, 7.92, 13.45],
        "ViTBase": [5.66, 3.96, 7.76],
        "ResNet18": [11.72, 12.04, 11.52],  # Added ResNet18 data
    },
}

simclr_std_main = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [4.44, 2.28, 0.21],
        "ViTBase": [3.37, 6.73, 5.45],
        "ResNet18": [2.91, 1.38, 0.97],  # Added ResNet18 std
    },
    "CIFAR10 KNN": {
        "ViTSmall": [2.79, 1.60, 0.69],
        "ViTBase": [2.75, 5.25, 4.87],
        "ResNet18": [1.70, 1.89, 1.08],  # Added ResNet18 std
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [2.55, 0.86, 0.74],
        "ViTBase": [3.92, 0.99, 5.02],
        "ResNet18": [1.40, 0.93, 0.49],  # Added ResNet18 std
    },
    "CIFAR100 KNN": {
        "ViTSmall": [0.20, 0.43, 0.61],
        "ViTBase": [1.38, 1.62, 2.19],
        "ResNet18": [0.95, 0.45, 0.74],  # Added ResNet18 std
    },
}

# Second plot data (New Method first, then ablations)
simclr_data_ablation = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [51.83, 50.52, 52.25, 48.31],
        "ViTBase": [44.37, 22.52, 39.94, 38.56],
        "ResNet18": [45.09, 45.60, 48.09, 48.85],  # Added ResNet18 data
    },
    "CIFAR10 KNN": {
        "ViTSmall": [40.29, 38.35, 38.82, 35.64],
        "ViTBase": [31.97, 18.02, 29.64, 27.89],
        "ResNet18": [35.92, 37.37, 40.10, 40.87],  # Added ResNet18 data
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [25.08, 23.23, 24.59, 21.15],
        "ViTBase": [18.08, 4.22, 15.07, 14.91],
        "ResNet18": [18.38, 18.65, 20.16, 20.85],  # Added ResNet18 data
    },
    "CIFAR100 KNN": {
        "ViTSmall": [13.45, 11.57, 12.44, 9.10],
        "ViTBase": [7.76, 3.16, 6.70, 6.65],
        "ResNet18": [11.52, 12.24, 13.32, 13.63],  # Added ResNet18 data
    },
}

simclr_std_ablation = {
    "CIFAR10 Linear Probe": {
        "ViTSmall": [0.21, 1.44, 1.08, 1.25],
        "ViTBase": [5.45, 3.29, 8.96, 8.81],
        "ResNet18": [0.97, 2.30, 0.92, 1.15],  # Added ResNet18 std
    },
    "CIFAR10 KNN": {
        "ViTSmall": [0.69, 1.12, 1.73, 1.42],
        "ViTBase": [4.87, 0.71, 6.00, 5.83],
        "ResNet18": [1.08, 2.20, 0.68, 1.40],  # Added ResNet18 std
    },
    "CIFAR100 Linear Probe": {
        "ViTSmall": [0.74, 1.88, 0.16, 0.91],
        "ViTBase": [5.02, 1.85, 5.99, 6.17],
        "ResNet18": [0.49, 1.35, 0.97, 1.10],  # Added ResNet18 std
    },
    "CIFAR100 KNN": {
        "ViTSmall": [0.61, 0.48, 1.33, 0.88],
        "ViTBase": [2.19, 0.49, 1.88, 2.29],
        "ResNet18": [0.74, 0.98, 0.08, 0.97],  # Added ResNet18 std
    },
}

# Colors and patterns
vitsmall_color = "#2ca02c"
vitbase_color = "#d62728"
resnet_color = "#1f77b4"  # Added ResNet18 color
patterns = ["", "//", "oo", "xx"]


def create_single_plot(ax, data, std_data, title, methods):
    bar_width = 0.12  # Reduced bar width to accommodate third model
    group_gap = 0.3

    n_methods = len(data["ViTSmall"])
    vitsmall_start = -(n_methods * bar_width + group_gap)
    vitbase_start = vitsmall_start + n_methods * bar_width + group_gap
    resnet_start = (
        vitbase_start + n_methods * bar_width + group_gap
    )  # Added ResNet18 position

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
        )

    ax.set_ylabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, pad=20)

    # Set x-ticks at the center of each group
    vitsmall_center = vitsmall_start + (n_methods * bar_width) / 2 - bar_width / 2
    vitbase_center = vitbase_start + (n_methods * bar_width) / 2 - bar_width / 2
    resnet_center = resnet_start + (n_methods * bar_width) / 2 - bar_width / 2
    ax.set_xticks([vitsmall_center, vitbase_center, resnet_center])
    ax.set_xticklabels(["ViTSmall", "ViTBase", "ResNet18"])


def create_plot(data, std_data, methods, title_suffix):
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"SimCLR Pre-training Results - {title_suffix}", fontsize=16)

    for i, (title, plot_data) in enumerate(data.items()):
        row = i // 2
        col = i % 2
        create_single_plot(axs[row, col], plot_data, std_data[title], title, methods)

    # Create legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#666666", alpha=0.6, hatch=p)
        for p in patterns
    ]
    fig.legend(handles, methods, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3, wspace=0.2)
    return fig


# Create main plot
main_methods = ["Balanced", "Imbalanced", "New Method"]
fig_main = create_plot(simclr_data_main, simclr_std_main, main_methods, "Main Methods")
plt.savefig("vit_comparison_main.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Create ablation plot with New Method first
ablation_methods = ["New Method", "OOD Aug.", "Uniform Gen.", "Stepwise"]
fig_ablation = create_plot(
    simclr_data_ablation, simclr_std_ablation, ablation_methods, "Ablation Methods"
)
plt.savefig("vit_comparison_ablation.pdf", format="pdf", bbox_inches="tight")
plt.show()
