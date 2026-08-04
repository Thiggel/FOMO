"""Create comparable per-run kNN sparsity histograms from isolated diagnostics."""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    series = []
    for spec in args.inputs:
        for path in glob.glob(spec, recursive=True):
            values = np.load(path)
            location = Path(path)
            series.append((f"{location.parents[4].name}/{location.parents[3].name}", values))
    if not series:
        raise RuntimeError("No isolated OOD diagnostics found")
    values = np.concatenate([item[1] for item in series])
    bins = np.linspace(np.quantile(values, .001), np.quantile(values, .999), 50)
    plt.figure(figsize=(7, 4))
    for label, item in series:
        plt.hist(item, bins=bins, density=True, histtype="step", linewidth=1.2, label=label)
    plt.xlabel("normalized kNN sparsity score"); plt.ylabel("density"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(output / "per_run_knn_histograms.png", dpi=220)


if __name__ == "__main__":
    main()
