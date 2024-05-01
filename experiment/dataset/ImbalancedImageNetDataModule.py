import os
import lightning.pytorch as L
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods, ImbalanceMethod


class ImbalancedImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_variant: ImageNetVariants = ImageNetVariants.ImageNet100,
        transform: Callable = None,
        splits: tuple[int, int] = (0.9, 0.1),
        batch_size: int = 32,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        resized_image_size: tuple[int, int] = (224, 224),
        checkpoint_filename: str = None,
    ):
        super().__init__()

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.num_classes
        ) = self._load_dataset(
            dataset_variant,
            imbalance_method,
            splits,
            checkpoint_filename
        )

        self.batch_size = batch_size

    def _load_dataset(
        self,
        dataset_variant: ImageNetVariants,
        imbalance_method: ImbalanceMethod,
        splits: tuple[float, float],
        checkpoint_filename: str
    ):
        dataset = ImbalancedImageNet(
            dataset_variant.value.path,
            imbalance_method=imbalance_method.value,
            checkpoint_filename=checkpoint_filename
        )

        num_classes = dataset.num_classes

        train_dataset, val_dataset = self._split_dataset(dataset, splits)

        test_dataset = ImbalancedImageNet(
            dataset_variant.value.path,
            split='test',
            imbalance_method=imbalance_method.value
        )

        return train_dataset, val_dataset, test_dataset, num_classes

    def _split_dataset(
        self,
        dataset: Dataset,
        splits: tuple[float, float],
    ) -> tuple[Dataset, Dataset, Dataset]:
        return random_split(
            dataset,
            self._get_splits(splits, dataset)
        )

    def _get_splits(
        self,
        splits: tuple[float, float],
        dataset: Dataset
    ) -> tuple[int, int, int]:
        size = len(dataset)

        train_size = int(splits[0] * size)
        val_size = size - train_size

        return train_size, val_size

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
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
