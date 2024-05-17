import os
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torch
import lightning.pytorch as L
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from experiment.utils.get_num_workers import get_num_workers


class CIFAR10KNNClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        k: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        (self.train_dataset, self.test_dataset) = self.get_datasets()

        self.model = model
        self.batch_size = batch_size
        self.max_epochs = 1

        self.knn = KNeighborsClassifier(n_neighbors=k)

    def get_datasets(self) -> tuple[Dataset, Dataset]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = CIFAR10(root="data", download=True, transform=transform)
        test_dataset = CIFAR10(
            root="data", train=False, download=True, transform=transform
        )

        return train_dataset, test_dataset

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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def extract_features(self, dataloader):
        features = []
        labels = []

        with torch.no_grad():
            for inputs, label in tqdm(dataloader, desc="Extracting features"):
                outputs = self.model.extract_features(inputs.to(self.device))
                features.append(outputs)
                labels.append(label)

        return torch.cat(features), torch.cat(labels)

    def fit_knn(self):
        train_loader = self.train_dataloader()
        train_features, train_labels = self.extract_features(train_loader)
        self.knn.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())

    def training_step(self, batch, batch_idx):
        self.fit_knn()
        self.trainer.should_stop = True
        return None

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch

        features = self.model.extract_features(inputs)
        predictions = self.knn.predict(features.cpu().numpy())
        accuracy = accuracy_score(targets.cpu().numpy(), predictions)
        print(targets, predictions, accuracy)

        self.log("cifar10_knn_test_accuracy", accuracy, prog_bar=True)

        return accuracy

    def configure_optimizers(self) -> optim.Optimizer:
        return None
