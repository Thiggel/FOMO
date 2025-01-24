import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import StanfordCars
from .BaseKNNClassifier import BaseKNNClassifier
from .StanfordCarsDataset import StanfordCarsDataset


class CarsKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
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
        self.train_dataset, self.test_dataset = random_split(
            base_dataset,
            [
                int(0.8 * len(base_dataset)),
                len(base_dataset) - int(0.8 * len(base_dataset)),
            ],
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
