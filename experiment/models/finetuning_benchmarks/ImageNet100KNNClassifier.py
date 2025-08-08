import os
from torch.utils.data import DataLoader
from .BaseKNNClassifier import BaseKNNClassifier
from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


class ImageNet100KNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        dm = ImbalancedImageNetDataModule(
            dataset_variant=ImageNetVariants.ImageNet100,
            transform=self.transform,
            imbalance_method=ImbalanceMethods.AllData,
        )
        dm.setup()
        self.train_dataset = dm.train_dataset
        self.test_dataset = dm.test_dataset

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
