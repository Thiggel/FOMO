from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
import torch
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods

def test_update_dataset():
    # Create an instance of ImbalancedImageNetDataModule
    datamodule = ImbalancedImageNetDataModule(dataset_variant=ImageNetVariants.ImageNetDummy,
                                              imbalance_method=ImbalanceMethods.NoImbalance)  
    
    initial_length = len(datamodule.train_dataset)

    # Add additional datapoints
    datamodule.dataset._save_additional_datapoint(torch.zeros(datamodule.train_dataset[0][0].shape), None)
    datamodule.train_dataset.indices.append(len(datamodule.dataset))
    datamodule.dataset._save_additional_datapoint(torch.ones(datamodule.train_dataset[0][0].shape), None)
    datamodule.train_dataset.indices.append(len(datamodule.dataset) + 1)

    updated_length = len(datamodule.train_dataset)

    assert updated_length == initial_length + 2

    # Check if the added datapoints are the extension of the train_dataset
    assert torch.equal(datamodule.train_dataset[initial_length][0], torch.zeros(datamodule.train_dataset[0].shape))
    assert datamodule.train_dataset[initial_length][1] == 0

    assert torch.equal(datamodule.train_dataset[initial_length + 1][0], torch.ones(datamodule.train_dataset[0].shape))
    assert datamodule.train_dataset[initial_length + 1][1] == 1