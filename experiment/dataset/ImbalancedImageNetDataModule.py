import lightning.pytorch as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from typing import Callable
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.ImbalancedImageNet import DummyImageNet, ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
from torchvision import transforms
import random
import os
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from experiment.utils.get_num_workers import get_num_workers


def worker_init_fn(worker_id):
    """Initialize workers with appropriate settings"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Set worker specific random seed
        torch.manual_seed(worker_info.seed)
        # Ensure each worker has its own CUDA stream
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())


class SafeDataLoader(DataLoader):
    """DataLoader with additional safety checks and cleanup"""

    def __init__(self, *args, **kwargs):
        if "persistent_workers" in kwargs and kwargs["num_workers"] == 0:
            kwargs["persistent_workers"] = False
        super().__init__(*args, **kwargs)

    def __iter__(self):
        try:
            return super().__iter__()
        except RuntimeError as e:
            if "torch_shm_manager" in str(e):
                print("Handling shared memory error...")
                # Attempt cleanup and retry
                if hasattr(self, "_iterator"):
                    del self._iterator
                torch.multiprocessing.set_sharing_strategy("file_system")
                return super().__iter__()
            raise


class ImbalancedImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        collate_fn: Callable = torch.utils.data._utils.collate.default_collate,
        dataset_variant: ImageNetVariants = ImageNetVariants.ImageNet100,
        transform: Callable = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        splits: tuple[int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        test_mode: bool = False,
    ):
        super().__init__()

        self.transform = transform
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        if test_mode:
            self.dataset = DummyImageNet(100, transform=self.transform)
        else:
            self.dataset = ImbalancedImageNet(
                dataset_variant.value.path,
                transform=self.transform,
                imbalance_method=imbalance_method,
                checkpoint_filename=checkpoint_filename,
            )

        self.num_classes = self.dataset.num_classes

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self._split_dataset(self.dataset, splits)

        # Initialize storage for original indices
        self.original_train_indices = None
        self.added_sample_indices = []

    def _split_dataset(
        self,
        dataset: Dataset,
        splits: tuple[float, float, float],
    ) -> tuple[Dataset, Dataset, Dataset]:
        return random_split(dataset, self._get_splits(dataset, splits))

    def _get_splits(
        self,
        dataset: Dataset,
        splits: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        size = len(dataset)

        train_size = int(splits[0] * size)
        val_size = int(splits[1] * size)
        test_size = size - train_size - val_size

        return train_size, val_size, test_size

    @property
    def num_workers(self) -> int:
        return min(12, get_num_workers())

    def set_dataloaders_none(self):
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def train_dataloader(self) -> DataLoader:
        self._train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        self._val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        self._test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return self._test_dataloader

    def collate(self, batch: list) -> tuple[list[Tensor], Tensor]:
        num_images = len(batch[0][0])

        outer_list = []
        for i in range(num_images):
            data = torch.stack(
                [
                    gg(
                        item[0][i].repeat(3, 1, 1)
                        if item[0][i].size(0) == 1
                        else item[0][i][:3]
                    )
                    for item in batch
                ]
            )

            outer_list.append(data)

        data = tuple(outer_list)
        labels = [item[1] for item in batch]

        stacked_labels = torch.tensor(labels)

        return data, stacked_labels

    def add_samples_by_index(self, indices_to_add: list[int]) -> None:
        """
        Add specific samples from the original dataset back into the training set.

        Args:
            indices_to_add (list[int]): List of indices from the original dataset to add back
        """
        if self.original_train_indices is None:
            # Store original indices first time this is called
            self.original_train_indices = self.train_dataset.indices.copy()

        # Get current training indices
        current_indices = list(self.train_dataset.indices)

        # Add new indices
        new_indices = current_indices + indices_to_add

        # Create new Subset with updated indices
        self.train_dataset = Subset(self.train_dataset.dataset, new_indices)

        # Store which samples we've added
        self.added_sample_indices.extend(indices_to_add)

        print(f"Dataset size after adding samples: {len(self.train_dataset)}")
        print(f"Total samples added so far: {len(self.added_sample_indices)}")

    def get_added_samples(self) -> list[int]:
        """
        Get list of indices that have been added back to the dataset.

        Returns:
            list[int]: List of added sample indices
        """
        return self.added_sample_indices
