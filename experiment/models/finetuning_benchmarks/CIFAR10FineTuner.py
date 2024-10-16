import os
import lightning.pytorch as L
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch

from experiment.utils.get_num_workers import get_num_workers


class CIFAR10FineTuner(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        lr: float = 0.01,
        output_size: int = 10,
        weight_decay=1e-3,
        max_epochs=25,
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = 32
        self.transform = transform

        self.save_hyperparameters(ignore=["model"])

        (self.train_dataset, _, _) = self.get_datasets()

        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_features = self.model.num_features

        self.probe = nn.Linear(self.num_features, output_size)

        (self.train_dataset, self.val_dataset, self.test_dataset) = self.get_datasets()

        self.loss = nn.CrossEntropyLoss()

    def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = CIFAR10(root="data", download=True, transform=self.transform)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        test_dataset = CIFAR10(
            root="data", train=False, download=True, transform=self.transform
        )

        return train_dataset, val_dataset, test_dataset

    @property
    def num_workers(self) -> int:
        return get_num_workers()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=1e-3,
            betas=(0.9, 0.95),
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.hparams.max_epochs * 0.6),
                int(self.hparams.max_epochs * 0.8),
            ],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        outputs = self.probe(outputs)

        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        outputs = self.probe(outputs)

        loss = self.loss(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        outputs = self.probe(outputs)

        loss = self.loss(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("cifar10_test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("cifar10_test_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return loss
