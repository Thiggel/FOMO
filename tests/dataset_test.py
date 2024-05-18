from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
import torch
from torchvision.utils import save_image
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
import os
from torchvision import transforms

def test_update_dataset():
    # Create an instance of ImbalancedImageNetDataModule
    datamodule = ImbalancedImageNetDataModule(dataset_variant=ImageNetVariants.ImageNetDummy,
                                              imbalance_method=ImbalanceMethods.NoImbalance,
                                              transform=transforms.ToTensor())  
    
    initial_length = len(datamodule.train_dataset)

    # Create a dummy image
    # Create a dummy image
    dummy_image = torch.zeros(datamodule.train_dataset[0][0].shape)

    # Save the dummy image as a file
    image_filename = "tests/dummy_image.jpg"
    save_image(dummy_image, image_filename)

    # Pass the image filename to _save_additional_datapoint
    datamodule.dataset._save_additional_datapoint(image_filename, None)
    datamodule.train_dataset.indices.append(len(datamodule.dataset))

    # Pass the image filename to _save_additional_datapoint
    datamodule.dataset._save_additional_datapoint(image_filename, None)
    datamodule.train_dataset.indices.append(len(datamodule.dataset)+1)

    # Check that the dataset has been updated
    assert torch.equal(datamodule.train_dataset[initial_length][0], torch.zeros(datamodule.train_dataset[0][0].shape))
    assert torch.equal(datamodule.train_dataset[initial_length+1][0], torch.zeros(datamodule.train_dataset[0][0].shape))

    # Remove the dummy image file
    os.remove(image_filename)