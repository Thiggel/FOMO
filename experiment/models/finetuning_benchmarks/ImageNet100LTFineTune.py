from torch.utils.data import DataLoader
from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.models.finetuning_benchmarks.TransferLearningBenchmark import (
    TransferLearningBenchmark,
)
from experiment.utils.set_seed import set_seed
from torch import nn
from torchvision import transforms


class ImageNet100LTFineTune(TransferLearningBenchmark):
    """Linear evaluation on the ImageNet-100-LT (long-tailed) dataset."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        seed: int = 42,
        **kwargs,
    ):
        self.seed = seed
        super().__init__(
            model=model,
            lr=lr,
            transform=transform,
            num_classes=100,
            *args,
            **kwargs,
        )
        self._log_per_class_accuracy = True
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        # Ensure the same long-tailed dataset split as used during training
        set_seed(self.seed)
        dm = ImbalancedDataModule(
            dataset_path="clane9/imagenet-100",
            imbalance_method=ImbalanceMethods.PowerLawImbalance,
            transform=self.transform,
        )
        return dm.train_dataset, dm.val_dataset, dm.test_dataset

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
