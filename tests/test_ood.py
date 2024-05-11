import numpy as np
import torch
from torch.utils.data import Dataset
from argparse import Namespace
import pytest
import logging
from torchvision import datasets, transforms
from experiment.ood.ood import OOD


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


class TestOODDummyDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        train_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        test_data = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]

        train_dataset = DummyDataset(train_data)

        test_dataset = DummyDataset(test_data)

        self.ood = OOD(
            args=Namespace(fe_batch_size=2, k=2, pct_ood=0.5, pct_train=0.5),
            train=train_dataset,
            test=test_dataset,
            feature_extractor=torch.nn.Identity(),
        )

    def test_extract_features(self):
        self.ood.extract_features()

        assert torch.equal(
            self.ood.train_features, torch.tensor([[1, 2, 3], [4, 5, 6]])
        )

        assert torch.equal(
            self.ood.test_features, torch.tensor([[7, 8, 9], [10, 11, 12]])
        )

    def test_ood(self):
        self.ood.train_features = torch.tensor(
            [[0, 0, 1], [0, 0, 0.6], [0, 0.1, 0.9], [0.1, 1.2, 0.01]]
        )

        self.ood.test_features = torch.tensor([[0, 0.1, 1.1], [7, 0, 0.1]])

        ood_indices, thresh = self.ood.ood(normalize=False)

        true_ood_indices = np.array([1])

        assert np.all(np.equal(ood_indices, true_ood_indices))

        assert len(ood_indices) == 1


class TestOODMNIST:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Load a small subset of MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Filter only two classes: 0 and 1
        train_data = []
        for img, label in train_dataset:
            if label in [0, 1]:
                train_data.append(img)
        test_data = []
        for img, label in test_dataset:
            if label in [0, 1]:
                test_data.append(img)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train = torch.utils.data.TensorDataset(
            torch.stack(train_data), torch.tensor([0, 1] * (len(train_data) // 2) + [0] * 1)
        )
        self.test = torch.utils.data.TensorDataset(
            torch.stack(test_data), torch.tensor([0, 1] * (len(test_data) // 2)  + [0] * 1)
        )

        self.ood = OOD(
            args=Namespace(fe_batch_size=2, k=2, pct_ood=0.5, pct_train=0.5),
            train=self.train,
            test=self.test,
            feature_extractor=torch.nn.Identity(),
        )

    def test_extract_features(self):
        self.ood.extract_features()

        assert torch.equal(
            self.ood.train_features,
            torch.stack([img for img, lab in self.train_dataset if lab in [0, 1]]),
        )

        assert torch.equal(
            self.ood.test_features,
            torch.stack([img for img, lab in self.test_dataset if lab in [0, 1]]),
        )

    def test_ood(self):
        self.ood.extract_features()

        # Artificially construct a test set with one class having a different class' datapoint
        test_data = torch.stack([img for img, lab in self.test_dataset if lab == 0])
        test_data[0] = torch.zeros_like(
            test_data[0]
        )  # Replacing a data point with all zeros from class 0
        
        self.ood.train_features = self.ood.train_features.view(self.ood.train_features.size(0), -1) # Flatten the data
        self.ood.test_features = test_data.view(test_data.size(0), -1) # Flatten the data

        ood_indices, thresh = self.ood.ood(normalize=False)

        

        # The first data point from the test set (which is now all zeros) should be classified as OOD
        assert ood_indices == [0]
