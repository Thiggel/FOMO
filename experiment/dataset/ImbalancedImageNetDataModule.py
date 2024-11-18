import os
import lightning.pytorch as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.ImbalancedImageNet import DummyImageNet, ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
from torchvision import transforms
from datasets import load_dataset
import random

from experiment.utils.get_num_workers import get_num_workers


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
        return min(24, get_num_workers() // 2)

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
                    (
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

    def add_n_samples_by_index(self, n):
        # get all the indices currently not in use
        unused_indices = list(
            set(range(len(self.dataset))) - set(self.train_dataset.indices)
        )

        # randomly sample n indices from the unused indices
        new_indices = random.sample(unused_indices, min(n, len(unused_indices)))

        # add the new indices to the indices list
        self.train_dataset.indices.extend(new_indices)
