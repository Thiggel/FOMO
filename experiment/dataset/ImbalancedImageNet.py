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
        streaming: bool = False,
    ):
        super().__init__()

        self.checkpoint_filename = checkpoint_filename
        self.transform = transform
        split = "train"
        self.additional_data_path = additional_data_path
        self.streaming = streaming

        print("Loading dataset", dataset_path)
        # Load the dataset, use streaming if specified
        self.dataset = load_dataset(
            dataset_path, split=split, trust_remote_code=True, streaming=streaming
        )

        # Get label information (works for both streaming and non-streaming datasets)
        self.labels = self.dataset.features["label"].names
        self.classes = self.dataset.features["label"].names
        self.num_classes = len(self.classes)
        self.imbalancedness = imbalance_method.value.impl(self.num_classes)

        # In streaming mode, we don't create indices for imbalance filtering
        # Instead, we use all samples from the dataset
        if not streaming:
            self.indices = self._load_or_create_indices()
        else:
            # For streaming datasets, we'll set a large value for dataset length
            # Actual length will be determined by the iterator in __getitem__
            self.dataset_length = 1000000  # Large number to represent "all samples"
            self.indices = list(range(self.dataset_length))  # Just for compatibility

        # Initialize image storage
        self.image_storage = ImageStorage(
            self.additional_data_path,
        )

        # Keep track of additional images per cycle
        self.additional_image_counts = self._load_or_create_image_counts()

        if not streaming:
            print("original length:", len(self.dataset))
            print("imbalanced length:", len(self))
        else:
            print("Using streaming dataset (no imbalance filtering)")

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
        if not self.streaming:
            print(f"Original indices length: {len(self.indices)}")
        print(f"Adding {num_images} generated images for cycle {cycle_idx}")
        print(f"Current additional_image_counts: {self.additional_image_counts}")

        self.additional_image_counts[cycle_idx] = {
            "count": num_images,
            "labels": labels,
        }
        self._save_image_counts()

        print(f"Updated additional_image_counts: {self.additional_image_counts}")
        total_additional = sum(
            data["count"] for data in self.additional_image_counts.values()
        )
        if not self.streaming:
            total = len(self.indices) + total_additional
            print(f"Calculated new total length: {total}")
        else:
            print(f"Added {total_additional} generated images")

    def __len__(self):
        """
        Total length is original indices plus all additional generated images.
        """
        total_additional = sum(
            cycle_data["count"] for cycle_data in self.additional_image_counts.values()
        )

        if self.streaming:
            # For streaming datasets, use a placeholder value plus additional images
            return self.dataset_length + total_additional
        else:
            return len(self.indices) + total_additional

    def _get_dataset_item_streaming(self, idx):
        """
        Get an item from the streaming dataset.
        For streaming datasets, we iterate through the dataset until we reach the desired index.
        """
        # Create an iterator if not already created
        if not hasattr(self, "_dataset_iter"):
            self._dataset_iter = iter(self.dataset)
            self._current_idx = 0

        # If the index is less than our current position, reset the iterator
        if idx < self._current_idx:
            self._dataset_iter = iter(self.dataset)
            self._current_idx = 0

        # Iterate until we reach the desired index
        try:
            while self._current_idx < idx:
                next(self._dataset_iter)
                self._current_idx += 1

            # Get the item at the desired index
            item = next(self._dataset_iter)
            self._current_idx += 1
            return item
        except StopIteration:
            # If we reach the end of the dataset, return None
            return None

    def _get_additional_image_info(self, idx: int) -> tuple[int, int, int]:
        """
        Convert an index beyond the original dataset into cycle index, image index, and label.

        Args:
            idx: Index into additional images

        Returns:
            tuple of (cycle_idx, image_idx, label)
        """
        if self.streaming:
            current_pos = self.dataset_length
        else:
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
        # Check if the index corresponds to the original dataset
        original_dataset_length = (
            self.dataset_length if self.streaming else len(self.indices)
        )

        if idx < original_dataset_length:
            # Get item from original dataset
            if self.streaming:
                # For streaming datasets, fetch by iterating through the dataset
                datapoint = self._get_dataset_item_streaming(idx)
                if datapoint is None:
                    # If we reached the end of the dataset, get an item from additional data
                    return self.__getitem__(original_dataset_length)
            else:
                # For non-streaming datasets, use the indices list
                dataset_idx = self.indices[idx]
                datapoint = self.dataset[dataset_idx]

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

        # Ensure the image is in RGB format
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def _create_indices(self) -> list[int]:
        """
        Create imbalanced dataset indices using GPU acceleration.
        """
        print("Creating dataset indices on GPU...")

        if not self.streaming:
            labels = torch.tensor(self.dataset["label"], device="cuda")
        else:
            # Use the loop approach for streaming datasets
            labels_list = []
            for i, item in enumerate(tqdm(self.dataset, desc="Extracting labels")):
                labels_list.append(item["label"])
            labels = torch.tensor(labels_list, device="cuda")

        # Convert to tensor and move to GPU
        labels = torch.tensor(labels_list, device="cuda")

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
        try:
            with open(self._indices_filename, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            print(
                f"No valid indices file found at {self._indices_filename}. Creating new indices."
            )
            return None

    def _load_or_create_indices(self):
        # Try loading existing indices first
        indices = self._load_indices_from_pickle()
        if indices is not None:
            return indices

        # If not available, create new indices
        return self._create_indices()
