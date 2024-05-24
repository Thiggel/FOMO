import os
import lightning.pytorch as L
from torch import nn
from torch.optim import AdamW, Optimizer
from torch import optim
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch

from experiment.utils.get_num_workers import get_num_workers
from experiment.models.finetuning_benchmarks.TestFineTuner import TestDataset


class SecondTestFineTuner(L.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # Dummy linear layer
        self.max_epochs = 10

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test1_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test1_val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test2_test_loss", 0.0)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(TestDataset(), batch_size=2)

    def val_dataloader(self):
        return DataLoader(TestDataset(), batch_size=2)

    def test_dataloader(self):
        return DataLoader(TestDataset(), batch_size=2)
