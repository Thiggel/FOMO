import torch
from torch.utils.data import Dataset
from experiment.ood.ood import OOD
import numpy as np
from argparse import Namespace
import pytest

import logging


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestOOD:
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

        true_ood_indices = torch.tensor([1])

        assert torch.equal(ood_indices, true_ood_indices)

        assert len(ood_indices) == 1
