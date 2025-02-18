import os
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from .BaseKNNClassifier import BaseKNNClassifier


class FlowersKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        base_path = os.getenv("BASE_CACHE_DIR")
        if stage == "fit" or stage is None:
            self.train_dataset = Flowers102(
                root=base_path + "/data",
                split="train",
                download=True,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = Flowers102(
                root=base_path + "/data",
                split="test",
                download=True,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )
