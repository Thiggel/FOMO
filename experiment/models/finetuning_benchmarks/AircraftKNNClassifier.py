import os
from torch.utils.data import DataLoader
from .BaseKNNClassifier import BaseKNNClassifier
from .AircraftFineTune import FGVCAircraft


class AircraftKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        base_path = os.getenv("BASE_CACHE_DIR")
        if stage == "fit" or stage is None:
            self.train_dataset = FGVCAircraft(
                root=base_path + "/data",
                split="train",
                download=True,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FGVCAircraft(
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
