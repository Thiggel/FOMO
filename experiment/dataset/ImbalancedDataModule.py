import os
import lightning.pytorch as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from typing import Callable
from experiment.dataset.ImbalancedDataset import ImbalancedDataset
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
from torchvision import transforms

from experiment.utils.get_num_workers import get_num_workers


class ImbalancedDataModule(L.LightningDataModule):
    def __init__(
        self,
        collate_fn: Callable = torch.utils.data._utils.collate.default_collate,
        dataset_path: str = "clane9/imagenet-100",
        dataset_name: str = None,
        split: str = "train+validation",
        x_key: str = "image",
        y_key: str = "label",
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
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        additional_data_path: str = "additional_data",
    ):
        super().__init__()

        self.transform = transform
        self.collate_fn = collate_fn
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.dataset = ImbalancedDataset(
            dataset_path,
            dataset_name=dataset_name,
            transform=self.transform,
            imbalance_method=imbalance_method,
            checkpoint_filename=checkpoint_filename,
            additional_data_path=additional_data_path,
            split=split,
            x_key=x_key,
            y_key=y_key,
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

    @property
    def _persistent_workers(self) -> bool:
        """Keep dataloader workers alive between epochs.

        Off by default because respawning was the safe choice while resumed
        runs were tearing down combined loaders after a checkpoint restore.
        The cost of that choice is large here: an epoch is a few hundred steps,
        so the pool is torn down and rebuilt hundreds of times per run, and
        each rebuild re-forks and re-imports before a single batch is produced.
        FOMO_PERSISTENT_WORKERS=1 opts a launcher into keeping them.
        """
        if self.num_workers <= 0:
            return False
        return os.environ.get("FOMO_PERSISTENT_WORKERS", "0") == "1"

    @property
    def _prefetch_factor(self):
        """Batches each worker runs ahead. ``None`` is required at 0 workers."""
        if self.num_workers <= 0:
            return None
        return int(os.environ.get("FOMO_PREFETCH_FACTOR", "4"))

    def set_dataloaders_none(self):
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def train_dataloader(self) -> DataLoader:
        self._train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        self._val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        self._test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            pin_memory=torch.cuda.is_available(),
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
