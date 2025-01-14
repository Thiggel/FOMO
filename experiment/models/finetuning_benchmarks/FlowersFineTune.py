import os
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
import warnings

from .TransferLearningBenchmark import TransferLearningBenchmark


class FlowersFineTune(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=102, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        base_dir = os.getenv("BASE_CACHE_DIR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = Flowers102(root=base_dir + "/data", download=True, transform=self.transform)

        train_size = int(
            0.7 * len(dataset)
        )  # Flowers102 is smaller, so we use more for training
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        return train_dataset, val_dataset, test_dataset
 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
