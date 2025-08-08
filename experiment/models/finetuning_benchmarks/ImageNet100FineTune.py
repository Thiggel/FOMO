from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .TransferLearningBenchmark import TransferLearningBenchmark
from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


class ImageNet100FineTune(TransferLearningBenchmark):
    def __init__(self, model: nn.Module, lr: float, transform: transforms.Compose, *args, **kwargs):
        super().__init__(
            model=model,
            lr=lr,
            transform=transform,
            num_classes=100,
            *args,
            **kwargs,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        dm = ImbalancedImageNetDataModule(
            dataset_variant=ImageNetVariants.ImageNet100,
            transform=self.transform,
            imbalance_method=ImbalanceMethods.AllData,
        )
        dm.setup()
        return dm.train_dataset, dm.val_dataset, dm.test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
        )
