import hashlib
import io
import os
from typing import Any, Optional

import torch
import requests
from PIL import Image
from torch.utils.data import Dataset
import pickle

from datasets import load_dataset
from datasets.features import ClassLabel
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)

from experiment.dataset.ImageStorage import ImageStorage


class ImbalancedDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        additional_data_path: str = "additional_data",
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        split: str = "train+validation",
        x_key: str = "image",
        y_key: Optional[str] = "label",
        checkpoint_filename: str = None,
        transform=None,
    ):
        super().__init__()

        self.checkpoint_filename = checkpoint_filename
        self.transform = transform
        self.additional_data_path = additional_data_path

        print("Loading dataset", dataset_path)
        self.dataset_path = dataset_path
        self.dataset = load_dataset(
            dataset_path,
            split=split,
            trust_remote_code=True
        )
        self.x_key = x_key
        self.y_key = y_key

        (
            self.labels,
            self.classes,
            self.num_classes,
            self.label_tensor,
        ) = self._initialize_label_metadata()
        self.imbalancedness = imbalance_method.value.impl(self.num_classes)
        self.indices = self._load_or_create_indices()

        # Initialize image storage
        self.image_storage = ImageStorage(
            self.additional_data_path,
        )

        self._image_cache_dir: Optional[str] = None
        self._download_timeout = 10

        # Keep track of additional images per cycle
        self.additional_image_counts = self._load_or_create_image_counts()

        print("original length:", len(self.dataset))
        print("imbalanced length:", len(self))

    def get_class_name(self, label: int) -> str:
        return self.labels[label]

    def _load_or_create_image_counts(self) -> dict:
        """Load or create a dictionary tracking number of images per cycle."""
        counts_file = os.path.join(
            os.environ["BASE_CACHE_DIR"],
            f"{self.additional_data_path}_image_counts.pkl",
        )
        try:
            with open(counts_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def _save_image_counts(self):
        """Save the current image counts to disk."""
        counts_file = os.path.join(
            os.environ["BASE_CACHE_DIR"],
            f"{self.additional_data_path}_image_counts.pkl",
        )
        with open(counts_file, "wb") as f:
            pickle.dump(self.additional_image_counts, f)

    def _initialize_label_metadata(
        self,
    ) -> tuple[list[str], list[str], int, torch.Tensor]:
        """Prepare label encodings and metadata for different label formats."""

        if self.y_key is None:
            labels = ["unlabeled"]
            num_classes = 1
            label_tensor = torch.zeros(len(self.dataset), dtype=torch.long)

            return labels, labels, num_classes, label_tensor

        feature = self.dataset.features.get(self.y_key)
        raw_labels = self.dataset[self.y_key]

        if isinstance(feature, ClassLabel):
            labels = feature.names
            num_classes = len(labels)

            if raw_labels and isinstance(raw_labels[0], str):
                encoded = [feature.str2int(label) for label in raw_labels]
            else:
                encoded = raw_labels

            label_tensor = torch.tensor(encoded, dtype=torch.long)

            return labels, labels, num_classes, label_tensor

        unique_values: list[Any] = list(dict.fromkeys(raw_labels))
        labels = [str(value) for value in unique_values]
        num_classes = len(labels)
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        encoded = [value_to_index[value] for value in raw_labels]
        label_tensor = torch.tensor(encoded, dtype=torch.long)

        return labels, labels, num_classes, label_tensor

    def add_generated_images(self, cycle_idx: int, num_images: int, labels: list[int]):
        """
        Update the count of generated images for a cycle.

        Args:
            cycle_idx: The training cycle index
            num_images: Number of new images generated
            labels: List of labels for the generated images
        """

        print(f"\nDEBUG:")
        print(f"Original indices length: {len(self.indices)}")
        print(f"Adding {num_images} generated images for cycle {cycle_idx}")
        print(f"Current additional_image_counts: {self.additional_image_counts}")

        self.additional_image_counts[cycle_idx] = {
            "count": num_images,
            "labels": labels,
        }
        self._save_image_counts()

        print(f"Updated additional_image_counts: {self.additional_image_counts}")
        total = len(self.indices) + sum(
            data["count"] for data in self.additional_image_counts.values()
        )
        print(f"Calculated new total length: {total}")

    def __len__(self):
        """
        Total length is original indices plus all additional generated images.
        """
        total_additional = sum(
            cycle_data["count"] for cycle_data in self.additional_image_counts.values()
        )
        return len(self.indices) + total_additional

    def _get_additional_image_info(self, idx: int) -> tuple[int, int, int]:
        """
        Convert an index beyond the original dataset into cycle index, image index, and label.

        Args:
            idx: Index into additional images

        Returns:
            tuple of (cycle_idx, image_idx, label)
        """
        current_pos = len(self.indices)

        for cycle_idx, cycle_data in sorted(self.additional_image_counts.items()):
            if idx < current_pos + cycle_data["count"]:
                image_idx = idx - current_pos
                label = cycle_data["labels"][image_idx]
                return cycle_idx, image_idx, label
            current_pos += cycle_data["count"]

        raise IndexError("Image index out of range")

    def __getitem__(self, idx) -> tuple:
        """
        Get an item from either the original dataset or generated images.
        """
        if idx < len(self.indices):
            # Get item from original dataset
            dataset_index = self.indices[idx]
            datapoint = self.dataset[dataset_index]
            image = self._load_image(datapoint)
            label = self.label_tensor[dataset_index].item()
        else:
            # Get generated image
            cycle_idx, image_idx, label = self._get_additional_image_info(idx)
            image = self.image_storage.load_image(cycle_idx, image_idx)

            assert image is not None, (
                f"Failed to load generated image at cycle {cycle_idx}, "
                f"index {image_idx}"
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _ensure_cache_dir(self) -> Optional[str]:
        if self._image_cache_dir is not None:
            return self._image_cache_dir

        base_cache_dir = os.environ.get("BASE_CACHE_DIR")
        if base_cache_dir is None:
            return None

        dataset_cache_dir = os.path.join(
            base_cache_dir,
            "hf_image_cache",
            self.dataset_path.replace("/", "_"),
        )
        os.makedirs(dataset_cache_dir, exist_ok=True)
        self._image_cache_dir = dataset_cache_dir
        return self._image_cache_dir

    def _cached_image_path(self, url: str) -> Optional[str]:
        cache_dir = self._ensure_cache_dir()
        if cache_dir is None:
            return None

        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".png"
        return os.path.join(cache_dir, filename)

    def _load_from_cache(self, path: str) -> Optional[Image.Image]:
        if not os.path.exists(path):
            return None

        with Image.open(path) as cached_image:
            return cached_image.copy()

    def _download_image(self, url: str) -> Image.Image:
        cache_path = self._cached_image_path(url)
        if cache_path:
            cached_image = self._load_from_cache(cache_path)
            if cached_image is not None:
                return cached_image

        response = requests.get(url, timeout=self._download_timeout)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image.save(cache_path)

            with Image.open(cache_path) as cached_image:
                return cached_image.copy()

        return image

    def _load_image(self, datapoint: dict) -> Image.Image:
        image_value = datapoint[self.x_key]

        if isinstance(image_value, Image.Image):
            return image_value

        if isinstance(image_value, dict):
            if "path" in image_value and image_value["path"]:
                with Image.open(image_value["path"]) as pil_image:
                    return pil_image.copy()
            if "bytes" in image_value and image_value["bytes"]:
                return Image.open(io.BytesIO(image_value["bytes"]))

        if isinstance(image_value, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_value))

        if isinstance(image_value, str):
            if image_value.startswith("http://") or image_value.startswith(
                "https://"
            ):
                return self._download_image(image_value)

            if os.path.exists(image_value):
                with Image.open(image_value) as pil_image:
                    return pil_image.copy()

        raise ValueError(
            f"Unsupported image type for key {self.x_key}: {type(image_value)}"
        )

    def _get_inverse_distribution(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        labels = self.label_tensor.to(device)
        imbalance_probs = self.imbalancedness.get_imbalance(labels)

        inverse = 1 - imbalance_probs

        return inverse, labels

    def _create_indices(self) -> list[int]:
        """
        Create imbalanced dataset indices using GPU acceleration.
        """
        print("Creating dataset indices on GPU...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load labels to target device tensor
        labels = self.label_tensor.to(device)

        # Get imbalance probabilities for all labels
        imbalance_probs = self.imbalancedness.get_imbalance(labels)

        # Generate random numbers for each sample
        random_numbers = torch.rand(len(labels), device=device)

        # Create mask to filter indices
        mask = imbalance_probs >= random_numbers
        selected_indices = torch.nonzero(mask, as_tuple=True)[0]

        # Convert to CPU and Python list
        selected_indices_list = selected_indices.cpu().tolist()

        # Save to pickle
        self._save_indices_to_pickle(selected_indices_list)

        remaining = len(selected_indices_list) / labels.size(0)
        print("\n--- Portion of dataset remaining: ", remaining, "\n ---")

        return selected_indices_list

    @property
    def _indices_filename(self):
        dataset_pickles = os.environ["BASE_CACHE_DIR"] + "/dataset_pickles"
        if not os.path.exists(dataset_pickles):
            os.makedirs(dataset_pickles, exist_ok=True)
        return f"{dataset_pickles}/{self.checkpoint_filename}_indices.pkl"

    def _save_indices_to_pickle(self, indices: list[int]):
        with open(self._indices_filename, "wb") as f:
            pickle.dump(indices, f)

    def _load_indices_from_pickle(self):
        with open(self._indices_filename, "rb") as f:
            return pickle.load(f)

    def _load_or_create_indices(self):
        return self._create_indices()
