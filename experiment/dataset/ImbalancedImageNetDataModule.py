import os
import lightning.pytorch as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
from PIL import Image
from torchvision import transforms


class ImbalancedImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        collate_fn: Callable,
        dataset_variant: ImageNetVariants = ImageNetVariants.ImageNet100,
        transform: Callable = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        splits: tuple[int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        resized_image_size: tuple[int, int] = (224, 224),
        checkpoint_filename: str = None,
    ):
        super().__init__()

        self.transform = transform
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.num_classes,
        ) = self._load_dataset(
            dataset_variant, imbalance_method, splits, checkpoint_filename
        )

    def _load_dataset(
        self,
        dataset_variant: ImageNetVariants,
        imbalance_method: ImbalanceMethod,
        splits: tuple[float, float],
        checkpoint_filename: str,
    ):
        dataset = ImbalancedImageNet(
            dataset_variant.value.path,
            transform=self.transform,
            imbalance_method=imbalance_method,
            checkpoint_filename=checkpoint_filename,
        )

        return self._split_dataset(dataset, splits) + [dataset.num_classes]

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
        return os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )
