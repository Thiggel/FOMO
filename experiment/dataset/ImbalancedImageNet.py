import os
from typing import TypedDict
from PIL import Image
from random import random
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from datasets import load_dataset
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
import h5py
import numpy as np
from PIL import Image
import io
import os
from typing import List, Tuple
from tqdm import tqdm


class ImageStorage:
    """
    Handles efficient storage of multiple images using HDF5 format.
    """

    def __init__(self, storage_path: str):
        """
        Initialize the storage with a path to the HDF5 file.

        Args:
            storage_path: Path where the HDF5 file will be stored
        """
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    def save_images(
        self, images: List[Image.Image], cycle_idx: int, labels: List[int]
    ) -> None:
        """
        Save a batch of PIL images to HDF5 storage.

        Args:
            images: List of PIL Image objects
            cycle_idx: Current training cycle index
            labels: List of corresponding labels
        """
        with h5py.File(self.storage_path, "a") as f:
            group_name = f"cycle_{cycle_idx}"
            if group_name not in f:
                cycle_group = f.create_group(group_name)
            else:
                cycle_group = f[group_name]

            for idx, (img, label) in enumerate(zip(images, labels)):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                image_name = f"image_{idx}"
                if image_name in cycle_group:
                    del cycle_group[image_name]
                    del cycle_group[f"{image_name}_label"]
                cycle_group.create_dataset(image_name, data=np.void(img_byte_arr))
                cycle_group.create_dataset(f"{image_name}_label", data=label)

    def load_image(self, cycle_idx: int, image_idx: int) -> Tuple[Image.Image, int]:
        """
        Load a single image and its label.

        Args:
            cycle_idx: The training cycle index
            image_idx: The index of the image within the cycle

        Returns:
            Tuple of (PIL Image, label)
        """
        with h5py.File(self.storage_path, "r") as f:
            group_name = f"cycle_{cycle_idx}"
            if group_name not in f:
                raise KeyError(f"Cycle {cycle_idx} not found")

            cycle_group = f[group_name]
            image_name = f"image_{image_idx}"

            img_bytes = cycle_group[image_name][()]
            img = Image.open(io.BytesIO(img_bytes.tobytes()))
            label = cycle_group[f"{image_name}_label"][()]

            return img, label

    def get_cycle_length(self, cycle_idx: int) -> int:
        """Get number of images in a cycle"""
        with h5py.File(self.storage_path, "r") as f:
            group_name = f"cycle_{cycle_idx}"
            if group_name not in f:
                return 0
            return len([k for k in f[group_name].keys() if not k.endswith("_label")])


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
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        transform=None,
    ):
        super().__init__()

        self.checkpoint_filename = checkpoint_filename
        self.transform = transform
        split = "train+validation"

        print("Loading dataset", dataset_path)
        self.dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)

        self.classes = self.dataset.features["label"].names
        self.num_classes = len(self.classes)
        self.imbalancedness = imbalance_method.value.impl(self.num_classes)
        self.indices = self._load_or_create_indices()

        # Initialize HDF5 storage
        storage_path = os.path.join(
            os.environ["BASE_CACHE_DIR"], f"{checkpoint_filename}_generated_images.h5"
        )
        self.image_storage = ImageStorage(storage_path)

        # Store cycle information
        self.cycle_lengths = {}
        self._update_cycle_lengths()

        print("original length:", len(self.dataset))
        print("imbalanced length:", len(self))

    def save_additional_datapoint(
        self, images: List[Image.Image], labels: List[int], cycle_idx: int
    ):
        """Save new generated images to HDF5 storage"""
        self.image_storage.save_images(images, cycle_idx, labels)
        self.cycle_lengths[cycle_idx] = len(images)

    def _create_or_load_additional_data(self):
        try:
            return self._load_additional_data_from_pickle()
        except FileNotFoundError:
            return []

    def _update_cycle_lengths(self):
        """Update the count of images in each cycle"""
        cycle_idx = 0
        while True:
            length = self.image_storage.get_cycle_length(cycle_idx)
            if length == 0:
                break
            self.cycle_lengths[cycle_idx] = length
            cycle_idx += 1

    @property
    def _additional_data_filename(self):
        dataset_pickles = os.environ["BASE_CACHE_DIR"] + "/dataset_pickles"

        if not os.path.exists(dataset_pickles):
            os.makedirs(dataset_pickles)

        return f"{dataset_pickles}/{self.checkpoint_filename}_additional_data.pkl"

    def _save_additional_data_to_pickle(self):
        with open(self._additional_data_filename, "wb") as f:
            pickle.dump(self.additional_data, f)

    def _load_additional_data_from_pickle(self):
        with open(self._additional_data_filename, "rb") as f:
            return pickle.load(f)

    def _create_indices(self) -> list[int]:
        """
        The imbalancedness class assigns an imbalance score
        between 0 and 1 to each class. Based on this, we randomly
        add a fraction of *1 - imbalance* of the samples of each class
        to our dataset.
        """
        indices = []

        for index, sample in tqdm(
            enumerate(self.dataset),
            total=len(self.dataset),
            desc="Making dataset imbalanced",
        ):
            if self.imbalancedness.get_imbalance(sample["label"]) < random():
                indices.append(index)

        self._save_indices_to_pickle(indices)

        return indices

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
        try:
            return self._load_indices_from_pickle()
        except FileNotFoundError:
            return self._create_indices()

    def __len__(self):
        """Total length is original indices plus all generated images"""
        return len(self.indices) + sum(self.cycle_lengths.values())

    def _load_additional_datapoint(self, idx) -> DataPoint:
        """
        We return the sample at the index in the additional data.
        """
        filename, label = self.additional_data[idx]

        return {"image": Image.open(filename), "label": label}

    def __getitem__(self, idx) -> DataPoint:
        """Get item from either original dataset or generated images"""
        if idx < len(self.indices):
            # Get from original dataset
            datapoint = self.dataset[self.indices[idx]]
        else:
            # Get from generated images
            adj_idx = idx - len(self.indices)
            cycle_idx = 0
            while adj_idx >= self.cycle_lengths.get(cycle_idx, 0):
                adj_idx -= self.cycle_lengths[cycle_idx]
                cycle_idx += 1

            img, label = self.image_storage.load_image(cycle_idx, adj_idx)
            datapoint = {"image": img, "label": label}

        datapoint["image"] = datapoint["image"].convert("RGB")

        if self.transform:
            datapoint["image"] = self.transform(datapoint["image"])

        return datapoint["image"], datapoint["label"]
