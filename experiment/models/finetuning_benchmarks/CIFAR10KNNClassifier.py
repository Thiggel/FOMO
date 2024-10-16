import os
import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
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
        self.use_deepspeed = False
        self.max_epochs = 1
        self.save_hyperparameters(ignore=["model"])
        self.transform = transform
        self.model = model
        self.batch_size = batch_size
        self.k = k
        self.knn = None  # We'll initialize this later

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                root="data", train=True, download=True, transform=self.transform
            )
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root="data", train=False, download=True, transform=self.transform
            )

    @property
    def num_workers(self) -> int:
        return get_num_workers()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers // 2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def extract_features(self, dataloader):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, label in tqdm(dataloader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                outputs = self.model.extract_features(inputs)
                features.append(outputs.cpu())
                labels.append(label)
        return torch.cat(features), torch.cat(labels)

    def on_train_start(self):
        # Fit KNN at the start of training
        train_loader = self.train_dataloader()
        train_features, train_labels = self.extract_features(train_loader)
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(train_features.numpy(), train_labels.numpy())

    def training_step(self, batch, batch_idx):
        # No need for actual training, return zero tensor to satisfy PyTorch Lightning
        return torch.tensor(0.0, requires_grad=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        features = self.model.extract_features(inputs)
        predictions = self.knn.predict(features.cpu().numpy())
        accuracy = accuracy_score(targets.cpu().numpy(), predictions)
        self.log("cifar10_knn_test_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return torch.tensor(accuracy)

    def configure_optimizers(self):
        # Return a dummy optimizer to satisfy DeepSpeed
        return optim.SGD(self.parameters(), lr=0)
