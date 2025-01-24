import os
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
from torchvision import transforms
from torchvision.datasets import StanfordCars
import warnings
from PIL import Image
import scipy.io as sio
from .TransferLearningBenchmark import TransferLearningBenchmark
from .StanfordCarsDataset import StanfordCarsDataset


class CarsFineTune(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=196, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        base_path = os.getenv("BASE_CACHE_DIR") + "/stanford_cars"
        base_dataset = StanfordCarsDataset(
            root_dir=base_path + "/cars_train",
            annotations_file=base_path + "/devkit/cars_train_annos.mat",
            transform=self.transform,
        )

        # always get the same train and test sets
        # because train and test is loaded independentl
        # and otherwise there is leakage
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            base_dataset,
            [
                int(0.8 * len(base_dataset)),
                int(0.1 * len(base_dataset)),
                len(base_dataset) - int(0.9 * len(base_dataset)),
            ],
            generator=generator,
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
