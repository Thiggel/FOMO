import os
import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable
from datasets import load_dataset
from dataset.ImageNetVariants import ImageNetVariants


class ImbalancedImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_variant: ImageNetVariants = ImageNetVariants.ImageNet100,
        transform: Callable = None,
        splits: tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        # TODO: define degree of imbalance in some way
        # (below is just a suggestion)
        imbalance: float = 0.1,
        resized_image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.dataset = self._load_dataset(dataset_variant, imbalance, splits)
        self.batch_size = batch_size

    def _load_dataset(
        self,
        dataset_variant: ImageNetVariants,
        imbalance: float,
        splits: tuple[float, float, float]
    ):
        dataset = load_dataset(dataset_variant.value.path)

        dataset = self._make_imbalanced(dataset, imbalance)

        return self._split_dataset(dataset, splits)

    def _make_imbalanced(self, dataset: Dataset, imbalance: float) -> Dataset:
        """
        Randomly pick a class from the dataset and remove a fracton of "imbalance"
        of its samples.
        """
        print(dataset)
        exit()

        return dataset

    def _split_dataset(
        self,
        dataset: Dataset,
        splits: tuple[float, float, float]
    ) -> tuple[Dataset, Dataset, Dataset]:
        return random_split(
            self.dataset,
            self._get_splits(splits)
        )

    def _get_splits(
        self,
        splits: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        size = len(self.dataset)

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
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
