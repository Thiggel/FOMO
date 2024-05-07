import torch
from torch.utils.data import Dataset
from experiment.ood.ood import extract_features, ood
import numpy as np

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
    train_features = np.array([[0, 0, 1], [0, 1, 0]])
    val_features = np.array([[0, 0, 0], [10, 5, 8]])
    
    # Call the ood function
    ood_scores, thresh = ood(train_features, val_features, 0.5)
    
    # Check output values
    assert ood_scores[0] == False 
    assert ood_scores[1] == True

    # Assert the expected output
    assert len(ood_scores) == 2
    assert isinstance(ood_scores[0], bool)
    assert isinstance(ood_scores[1], bool)
