import os
from tqdm import tqdm
import torch
from typing import TypedDict
from PIL import Image
from torch.utils.data import Dataset
import pickle

from datasets import load_dataset
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)

from experiment.dataset.ImageStorage import ImageStorage


class DataPoint(TypedDict):
    image: Image.Image
    label: int


class DummyImageNet(Dataset):
    def __init__(self, num_classes: int, transform=None):
        super().__init__()

        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return 100

    def __getitem__(self, _) -> tuple[Image.Image, int]:
        datapoint: DataPoint = {"image": Image.new("RGB", (224, 224)), "label": 0}

        if self.transform:
            datapoint["image"] = self.transform(datapoint["image"])

        return datapoint["image"], datapoint["label"]


class ImbalancedImageNet(Dataset):
    def __init__(
        self,
        dataset_path: str,
        additional_data_path: str = "additional_data",
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        transform=None,
    ):
        super().__init__()

        self.checkpoint_filename = checkpoint_filename
        self.transform = transform
        split = "train+validation"
        self.additional_data_path = additional_data_path

        print("Loading dataset", dataset_path)
        self.dataset = load_dataset(
            dataset_path,
            split=split,
            trust_remote_code=True,
        )
        self.labels = self.dataset.features["label"].names

        self.classes = self.dataset.features["label"].names
        self.num_classes = len(self.classes)
        self.imbalancedness = imbalance_method.value.impl(self.num_classes)
        self.indices = self._load_or_create_indices()

        # Initialize image storage
        self.image_storage = ImageStorage(
            self.additional_data_path,
        )

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
            datapoint = self.dataset[self.indices[idx]]
            image = datapoint["image"]
            label = datapoint["label"]
        else:
            # Get generated image
            cycle_idx, image_idx, label = self._get_additional_image_info(idx)
            image = self.image_storage.load_image(cycle_idx, image_idx)

            assert image is not None, (
                f"Failed to load generated image at cycle {cycle_idx}, "
                f"index {image_idx}"
            )

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_inverse_distribution(self):
        labels = torch.tensor(self.dataset["label"], device="cuda")
        imbalance_probs = self.imbalancedness.get_imbalance(labels)

        inverse = 1 - imbalance_probs

        return inverse, labels

    def _create_indices(self) -> list[int]:
        """
        Create imbalanced dataset indices using GPU acceleration.
        """
        print("Creating dataset indices on GPU...")

        # Load labels to a GPU tensor
        labels = torch.tensor(self.dataset["label"], device="cuda")

        # Get imbalance probabilities for all labels
        imbalance_probs = self.imbalancedness.get_imbalance(labels)

        # Generate random numbers for each sample
        random_numbers = torch.rand(len(labels), device="cuda")

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
        """Load previously generated indices or create them if necessary."""
        if self.checkpoint_filename and os.path.exists(self._indices_filename):
            return self._load_indices_from_pickle()

        indices = self._create_indices()
        return indices
