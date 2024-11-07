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
import shutil
import h5py
import numpy as np
from PIL import Image
import io
import os
from typing import List, Tuple
from tqdm import tqdm
import filelock
import torch.distributed as dist


class DistributedImageStorage:
    """
    Handles efficient storage of images with separate paths for reading and writing,
    optimized for distributed GPU training, with chunked storage per cycle.
    """

    def __init__(self, storage_path: str, max_images_per_file: int = 100):
        """
        Initialize the storage system.

        Args:
            storage_path: Base path for storage
            max_images_per_file: Maximum number of images per HDF5 file
        """
        self.base_path = storage_path
        self.lock_path = f"{storage_path}.lock"
        self.cache_dir = f"{storage_path}_cache"
        self.max_images_per_file = max_images_per_file

        # Create directories
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cycle_file(self, cycle_idx: int, file_idx: int, mode: str):
        """Helper to get the file path and handle for a given cycle and file chunk."""
        cycle_path = f"{self.base_path}_cycle_{cycle_idx}_part_{file_idx}.h5"
        return h5py.File(cycle_path, mode), cycle_path

    def _get_file_lock(self):
        """Get a file lock for thread-safe operations"""
        return filelock.FileLock(self.lock_path, timeout=30)

    def save_images(
        self, images: List[Image.Image], cycle_idx: int, labels: List[int]
    ) -> None:
        """
        Save new images in chunks within a cycle-specific HDF5 structure.
        Only rank 0 performs the write operation.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # Calculate starting file and image indices for the current cycle
        num_images = self.get_cycle_length(cycle_idx)
        start_file_idx = num_images // self.max_images_per_file
        start_image_idx = num_images % self.max_images_per_file

        with self._get_file_lock():
            file_idx = start_file_idx
            image_idx = start_image_idx

            for img, label in zip(images, labels):
                if image_idx == 0 or image_idx >= self.max_images_per_file:
                    # Open new file if at the start or when max_images_per_file is reached
                    image_file, _ = self._get_cycle_file(cycle_idx, file_idx, "a")
                    file_idx += 1
                    image_idx = 0
                else:
                    # Reopen the file for current chunk
                    image_file, _ = self._get_cycle_file(cycle_idx, file_idx - 1, "a")

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_data = np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8)

                image_name = f"image_{image_idx}"
                if image_name in image_file:
                    del image_file[image_name]
                    del image_file[f"{image_name}_label"]

                image_file.create_dataset(
                    image_name,
                    data=img_data,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                image_file.create_dataset(f"{image_name}_label", data=label)

                image_idx += 1
                image_file.close()

        # Ensure other processes are aware of the updated files
        if dist.is_initialized():
            dist.barrier()

    def load_image(self, cycle_idx: int, image_idx: int) -> Tuple[Image.Image, int]:
        """
        Load a single image and its label from the appropriate HDF5 chunk file or cache.
        """
        cache_key = f"{cycle_idx}_{image_idx}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npz")

        # Try loading from cache first
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                img_data = data["image"]
                label = int(data["label"])
                return Image.open(io.BytesIO(img_data)), label
            except:
                if os.path.exists(cache_path):
                    os.remove(cache_path)

        # Determine the file and local index within the file
        file_idx = image_idx // self.max_images_per_file
        local_idx = image_idx % self.max_images_per_file

        with self._get_cycle_file(cycle_idx, file_idx, "r") as f:
            image_name = f"image_{local_idx}"

            if image_name not in f:
                raise KeyError(f"Image {image_idx} not found in cycle {cycle_idx}")

            img_data = f[image_name][()].tobytes()
            label = f[f"{image_name}_label"][()]

            # Cache the result
            np.savez_compressed(cache_path, image=img_data, label=label)

            return Image.open(io.BytesIO(img_data)), label

    def get_cycle_length(self, cycle_idx: int) -> int:
        """Calculate the total number of images stored across all HDF5 chunk files for a cycle."""
        total_images = 0
        file_idx = 0

        while True:
            try:
                with self._get_cycle_file(cycle_idx, file_idx, "r") as f:
                    total_images += len(
                        [k for k in f.keys() if not k.endswith("_label")]
                    )
                file_idx += 1
            except OSError:
                # Break if no more files exist
                break

        return total_images

    def delete_cycle(self, cycle_idx: int) -> None:
        """Delete all HDF5 files for a specific cycle to free up space."""
        file_idx = 0
        while True:
            _, cycle_path = self._get_cycle_file(cycle_idx, file_idx, "r")
            if os.path.exists(cycle_path):
                os.remove(cycle_path)
                file_idx += 1
            else:
                break


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
        self.image_storage = DistributedImageStorage(storage_path)

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
