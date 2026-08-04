import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import StanfordCars
from .BaseKNNClassifier import BaseKNNClassifier
from .StanfordCarsDataset import (
    StanfordCarsDataset,
    HuggingFaceStanfordCarsDataset,
)
from .DatasetRoots import get_stanford_cars_root


class CarsKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        base_path = os.getenv("STANFORD_CARS_ROOT")
        train_root = os.path.join(base_path, "cars_train") if base_path else ""
        annotations_file = (
            os.path.join(base_path, "devkit", "cars_train_annos.mat")
            if base_path
            else ""
        )
        if os.path.isdir(train_root) and os.path.isfile(annotations_file):
            base_dataset = StanfordCarsDataset(
                root_dir=train_root,
                annotations_file=annotations_file,
                transform=self.transform,
            )
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.test_dataset = random_split(
                base_dataset,
                [
                    int(0.8 * len(base_dataset)),
                    len(base_dataset) - int(0.8 * len(base_dataset)),
                ],
                generator=generator,
            )
        else:
            print(
                "Stanford Cars legacy files are unavailable; using "
                "tanganke/stanford_cars from Hugging Face."
            )
            self.train_dataset = HuggingFaceStanfordCarsDataset(
                split="train", transform=self.transform
            )
            self.test_dataset = HuggingFaceStanfordCarsDataset(
                split="test", transform=self.transform
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
