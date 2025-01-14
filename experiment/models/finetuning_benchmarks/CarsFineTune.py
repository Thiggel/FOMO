import os
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
from torchvision import transforms
from torchvision.datasets import StanfordCars
import warnings
from PIL import Image
import scipy.io as sio
from .TransferLearningBenchmark import TransferLearningBenchmark

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
            model=model, lr=lr, transform=None, num_classes=196, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()
        self.transform = self.get_transform()

    def get_datasets(self):
        base_path = os.getenv("BASE_CACHE_DIR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_dataset = StanfordCars(
                root=base_path,
                download=False,
                transform=self.transform
            )
            test_dataset = StanfordCars(
                root=base_path,
                download=False,
                transform=self.transform
            )

        # Split train into train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
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

    def calculate_normalization(self, crop_size):
        temp_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
        
        # Save original transform
        original_transform = self.transform
        self.transform = temp_transform
        
        loader = DataLoader(
            self,
            batch_size=128,
            num_workers=4,
            shuffle=False
        )
        
        mean = 0.0
        std = 0.0
        total_images = 0
        
        # Calculate mean
        for images, _ in tqdm(loader, desc="Calculating mean"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            total_images += batch_samples
        
        mean /= total_images
        
        # Calculate std
        for images, _ in tqdm(loader, desc="Calculating std"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
        
        std = torch.sqrt(std / total_images)
        
        # Restore original transform
        self.transform = original_transform
        
        return mean.tolist(), std.tolist()
