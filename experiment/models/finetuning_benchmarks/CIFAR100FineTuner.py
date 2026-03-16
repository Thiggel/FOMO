from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import warnings

from .TransferLearningBenchmark import TransferLearningBenchmark
from .DatasetRoots import get_data_root, allow_data_downloads


class CIFAR100FineTuner(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=100, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        data_root = get_data_root()
        allow_downloads = allow_data_downloads()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = CIFAR100(
                root=data_root,
                download=allow_downloads,
                transform=self.transform,
            )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        test_dataset = CIFAR100(
            root=data_root,
            train=False,
            download=allow_downloads,
            transform=self.transform,
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
