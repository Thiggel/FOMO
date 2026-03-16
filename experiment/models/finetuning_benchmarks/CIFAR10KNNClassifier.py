from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from .BaseKNNClassifier import BaseKNNClassifier
from .DatasetRoots import get_data_root, allow_data_downloads


class CIFAR10KNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        data_root = get_data_root()
        allow_downloads = allow_data_downloads()
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                root=data_root,
                train=True,
                download=allow_downloads,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root=data_root,
                train=False,
                download=allow_downloads,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
