import os
import pytorch_lightning as L
from torch import nn
from torch.optim import Adam, Optimizer
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch


class CIFAR10FineTuner(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        output_size: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        (self.train_dataset, self.val_dataset, self.test_dataset) = self.get_datasets()

        self.model = model
        self.max_epochs = 10

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        self.loss = nn.CrossEntropyLoss()

    def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = CIFAR10(root="data", download=True, transform=transform)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        test_dataset = CIFAR10(
            root="data", train=False, download=True, transform=transform
        )

        return train_dataset, val_dataset, test_dataset

    @property
    def num_workers(self) -> int:
        return os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("cifar10_test_loss", loss)
        self.log("cifar10_test_accuracy", accuracy)
        return loss
