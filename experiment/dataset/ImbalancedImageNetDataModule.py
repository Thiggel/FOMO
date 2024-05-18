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
        collate_fn: Callable = torch.utils.data._utils.collate.default_collate,
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
        test_mode: bool = False,
    ):
        super().__init__()

        self.transform = transform
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        self.dataset = ImbalancedImageNet(
            dataset_variant.value.path,
            transform=self.transform,
            imbalance_method=imbalance_method,
            checkpoint_filename=checkpoint_filename,
            test_mode=test_mode,
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
        return os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #persistent_workers=True,
            collate_fn=self.collate_fn,
        )

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

    def update_dataset(self, aug_path):
        # TODO: add the real labels, not dummy ones
        images = os.listdir(aug_path)
        for image in images:
            self.dataset._save_additional_datapoint(image, None)
            self.train_dataset.indices.append(len(self.dataset) - 1)
