"""Utility to visualize generated augmentation samples across cycles.

This script loads the bookkeeping pickle produced during training to
identify how many images were generated per cycle. It then reuses the
HDF5-based storage managed by :class:`experiment.dataset.ImageStorage`
to retrieve a handful of representative augmentations for each cycle and
renders them into a matplotlib grid that is persisted to disk.  The
resulting figure can be used in reports without depending on WandB.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiment.dataset.ImageStorage import ImageStorage


CountDict = Dict[int, Dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize generated augmentation samples per cycle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--counts-file",
        type=Path,
        required=True,
        help="Path to the *_image_counts.pkl file generated during training",
    )
    parser.add_argument(
        "--additional-data-path",
        type=Path,
        default=None,
        help=(
            "Base directory that stores generated samples in HDF5 format. "
            "If omitted, it is inferred from the counts-file name."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_samples_overview.png"),
        help="Where to store the resulting figure",
    )
    parser.add_argument(
        "--images-per-cycle",
        type=int,
        default=4,
        help="Number of generated samples to display per cycle",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of cycle indices to visualize. Defaults to all cycles",
    )

    return parser.parse_args()


def infer_additional_data_path(counts_file: Path) -> Path:
    stem = counts_file.name
    suffix = "_image_counts.pkl"
    if not stem.endswith(suffix):
        raise ValueError(
            "Cannot infer additional-data path from counts file name; "
            "please provide --additional-data-path explicitly."
        )

    base_name = stem[: -len(suffix)]
    return counts_file.with_name(base_name)


def load_counts(counts_file: Path) -> CountDict:
    with counts_file.open("rb") as handle:
        counts: CountDict = pickle.load(handle)

    if not isinstance(counts, dict):
        raise TypeError(
            "Counts file did not contain the expected dictionary structure."
        )

    return counts


def sorted_cycles(counts: CountDict, subset: Iterable[int] | None) -> List[int]:
    available = sorted(int(k) for k in counts.keys())
    if subset is None:
        return available

    subset_set = set(subset)
    missing = subset_set - set(available)
    if missing:
        raise ValueError(f"Requested cycles {sorted(missing)} are not present in counts file")

    return [cycle for cycle in available if cycle in subset_set]


def ensure_output_dir(path: Path) -> None:
    output_dir = path.parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def pil_to_array(image: Image.Image) -> np.ndarray:
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    return np.asarray(image)


def collect_images_for_cycle(
    storage: ImageStorage, cycle_idx: int, count: int, limit: int
) -> List[Tuple[int, Image.Image]]:
    num_samples = min(count, limit)
    images: List[Tuple[int, Image.Image]] = []

    for global_idx in range(num_samples):
        loaded = storage.load_image(cycle_idx, global_idx)
        if loaded is None:
            continue
        images.append((global_idx, loaded))

    return images


def plot_cycles(
    cycle_images: List[Tuple[int, List[Tuple[int, Image.Image]]]],
    images_per_cycle: int,
    output_path: Path,
) -> None:
    if not cycle_images:
        raise RuntimeError("No images were collected; nothing to plot.")

    num_cycles = len(cycle_images)
    fig, axes = plt.subplots(
        nrows=num_cycles,
        ncols=images_per_cycle,
        figsize=(images_per_cycle * 3, num_cycles * 3),
        squeeze=False,
    )

    for row_idx, (cycle_idx, images) in enumerate(cycle_images):
        for col_idx in range(images_per_cycle):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col_idx < len(images):
                global_idx, image = images[col_idx]
                ax.imshow(pil_to_array(image))
                ax.set_title(f"Cycle {cycle_idx} · idx {global_idx}", fontsize=10)
            else:
                ax.set_title(f"Cycle {cycle_idx} · (empty)", fontsize=10)

    fig.tight_layout()
    ensure_output_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    counts_file = args.counts_file.expanduser().resolve()
    if not counts_file.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}")

    additional_path = (
        args.additional_data_path.expanduser().resolve()
        if args.additional_data_path is not None
        else infer_additional_data_path(counts_file)
    )

    if not additional_path.exists():
        raise FileNotFoundError(
            "Generated samples directory does not exist. "
            f"Expected to find it at: {additional_path}"
        )

    counts = load_counts(counts_file)
    cycles = sorted_cycles(counts, args.cycles)

    if not cycles:
        raise RuntimeError("No cycles available to visualize after filtering")

    storage = ImageStorage(str(additional_path))
    cycle_images: List[Tuple[int, List[Tuple[int, Image.Image]]]] = []

    for cycle_idx in cycles:
        cycle_info = counts.get(cycle_idx)
        if not cycle_info:
            continue

        count = int(cycle_info.get("count", 0))
        if count <= 0:
            continue

        images = collect_images_for_cycle(storage, cycle_idx, count, args.images_per_cycle)
        if images:
            cycle_images.append((cycle_idx, images))

    if not cycle_images:
        raise RuntimeError(
            "Could not retrieve any images. Confirm that the counts file and "
            "additional data directory correspond to the same run."
        )

    plot_cycles(cycle_images, args.images_per_cycle, args.output.expanduser())
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
