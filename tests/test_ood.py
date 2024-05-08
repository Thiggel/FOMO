import torch
from torch.utils.data import Dataset
from experiment.ood.ood import extract_features, ood
import numpy as np

import logging

class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_extract_features():
    # Create dummy datasets
    train_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    val_data = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
    train_dataset = DummyDataset(train_data)
    val_dataset = DummyDataset(val_data)
    
    # Create a dummy feature extractor
    class DummyFeatureExtractor(torch.nn.Module):
        def forward(self, x):
            return x
    
    feature_extractor = DummyFeatureExtractor()
    
    # Call the extract_features function
    train_features, val_features = extract_features(train_dataset, val_dataset, feature_extractor)
    
    # Assert the expected output
    assert np.all(np.equal(train_features, np.array([[1, 2, 3], [4, 5, 6]])))
    assert np.all(np.equal(val_features, np.array([[7, 8, 9], [10, 11, 12]])))
    
def test_ood():
    # Create dummy features
    train_features = np.array([[0, 0, 1], [0, 0, 0.6], [0, 0.1, 0.9], [0.1, 1.2, 0.01]])
    val_features = np.array([[0, 0.1, 1.1], [7, 0, 0.1]])
    
    # Call the ood function
    ood_scores, thresh = ood(train_features, val_features, K=2, pct_ood=0.5)
    true_ood_scores = np.array([False, True])
    # Check output values
    assert np.all(np.equal(ood_scores, true_ood_scores))

    # Assert the expected output
    assert len(ood_scores) == 2
