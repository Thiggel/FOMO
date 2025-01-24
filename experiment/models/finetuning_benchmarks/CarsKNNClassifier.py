import os
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars
from .BaseKNNClassifier import BaseKNNClassifier
from .StanfordCarsDataset import StanfordCarsDataset


class CarsKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        base_path = os.getenv("BASE_CACHE_DIR") + "/stanford_cars"

        if stage == "fit" or stage is None:
            self.train_dataset = StanfordCarsDataset(
                root=base_path + "/cars_train",
                annotations_file=base_path + "/devkit/cars_train_annos.mat",
                transform=self.transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = StanfordCarsDataset(
                root=base_path + "/cars_test",
                annotations_file=base_path + "/devkit/cars_test_annos.mat",
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
