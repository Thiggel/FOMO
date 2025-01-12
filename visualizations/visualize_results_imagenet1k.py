import matplotlib.pyplot as plt

# Data dictionaries
simclr_data_main = {
    "CIFAR10 Linear Probe": {
        "ViTBase": [46.57, 47.34, 51.85],
        "ResNet101": [36.61, 33.34, 37.71],
    },
    "CIFAR10 KNN": {
        "ViTBase": [36.52, 36.56, 36.80],
        "ResNet101": [27.14, 25.61, 28.28],
    },
    "CIFAR100 Linear Probe": {
        "ViTBase": [21.28, 21.76, 26.08],
        "ResNet101": [10.31, 8.76, 13.49],
    },
    "CIFAR100 KNN": {"ViTBase": [12.83, 13.46, 13.16], "ResNet101": [6.80, 6.26, 7.75]},
}

simclr_std_main = {
    "CIFAR10 Linear Probe": {
        "ViTBase": [3.46, 4.98, 0.03],
        "ResNet101": [2.39, 0.19, 1.36],
    },
    "CIFAR10 KNN": {"ViTBase": [0.19, 0.21, 0.11], "ResNet101": [0.09, 0.47, 1.87]},
    "CIFAR100 Linear Probe": {
        "ViTBase": [3.04, 4.15, 0.14],
        "ResNet101": [2.54, 1.04, 0.70],
    },
    "CIFAR100 KNN": {"ViTBase": [0.14, 0.01, 0.25], "ResNet101": [0.32, 0.14, 1.40]},
}

# Colors and patterns
vitbase_color = "#d62728"  # Red
resnet_color = "#1f77b4"  # Blue
patterns = ["", "//", "oo"]  # Patterns for different methods


def create_single_plot(ax, data, std_data, title, methods):
    bar_width = 0.3
    group_gap = 0.3

    n_methods = len(data["ViTBase"])
    vitbase_start = -(n_methods * bar_width + group_gap / 2)
    resnet_start = group_gap / 2

    for i in range(n_methods):
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

        # ResNet101 bars
        resnet_pos = resnet_start + i * bar_width
        resnet_val = data["ResNet101"][i]
        resnet_std = std_data["ResNet101"][i]

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

    ax.set_ylabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, pad=40)

    # Set x-ticks at the center of each group
    vitbase_center = vitbase_start + (n_methods * bar_width) / 2 - bar_width / 2
    resnet_center = resnet_start + (n_methods * bar_width) / 2 - bar_width / 2
    ax.set_xticks([vitbase_center, resnet_center])
    ax.set_xticklabels(["ViT-Base (DINO)", "ResNet101 (SimCLR)"])


def create_plot(data, std_data, methods, title_suffix):
    # Increase the overall font size for all text in the plots
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
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
        ncol=3,
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
plt.savefig("comparison-plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
