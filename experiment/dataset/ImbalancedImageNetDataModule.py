import os
import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset, ConcatDataset
from typing import Callable
from datasets import load_dataset, concatenate_datasets
from dataset.ImageNetVariants import ImageNetVariants
from torchvision import transforms
from PIL import Image
import torch

# Define default transformations for ImageNet
default_transforms = transforms.Compose([
    transforms.Resize(256),         # Resize the input image to 256x256
    transforms.CenterCrop(224),     # Crop the center 224x224 region
    transforms.ToTensor(),        # Convert PIL image to tensor (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize(           # Normalize with ImageNet statistics
        mean=[0.485, 0.456, 0.406],  # Mean of ImageNet dataset
        std=[0.229, 0.224, 0.225]     # Standard deviation of ImageNet dataset
    ),
])


class ImbalancedImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_variant: ImageNetVariants = ImageNetVariants.ImageNet100,
        transform: Callable = default_transforms,
        splits: tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        # TODO: define degree of imbalance in some way
        # (below is just a suggestion)
        imbalance: float = 0.1,
        resized_image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.transforms = transform

        self.train_dataset, self.val_dataset, self.test_dataset = self._load_dataset(dataset_variant, imbalance, splits)
        self.batch_size = batch_size
        print(len(self.train_dataset))
        print(len(self.val_dataset))
        print(len(self.test_dataset))

    def _load_dataset(
        self,
        dataset_variant: ImageNetVariants,
        imbalance: float,
        splits: tuple[float, float, float]
    ):
        dataset_train = HuggingFaceDatasetWrapper(dataset_variant.value.path, split='train', transform=self.transforms)
        dataset_test = HuggingFaceDatasetWrapper(dataset_variant.value.path, split='validation', transform=self.transforms)
        # Combine train and validation datasets
        dataset = ConcatDataset([dataset_train, dataset_test])

        self._make_imbalanced(dataset, imbalance)
        return self._split_dataset(dataset, splits)


    def _make_imbalanced(self, dataset: Dataset, imbalance: float) -> Dataset:
        """
        Randomly pick a class from the dataset and remove a fracton of "imbalance"
        of its samples.
        """
        print(dataset)
        #exit()

        return dataset

    def _split_dataset(
        self,
        dataset: Dataset,
        splits: tuple[float, float, float]
    ) -> tuple[Dataset, Dataset, Dataset]:
        return random_split(
            dataset,
            self._get_splits(dataset, splits)
        )

    def _get_splits(
        self,
        dataset: Dataset,
        splits: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        size = len(dataset)

        train_size = int(splits[0] * size)
        val_size = int(splits[1] * size)
        test_size = size - train_size - val_size

        return train_size, val_size, test_size

    @property
    def num_workers(self) -> int:
        return os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn = my_collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn = my_collate
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn = my_collate
        )

class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, dataset_name, split='train', transform=None):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # Assuming the image data is stored directly in 'image'
        label = item['label']  # Assuming the label is stored in 'label' field

        if self.transform:
            image = self.transform(image)
        return image, label

def my_collate(batch):
    num_images = len(batch[0][0])

    outer_list = []
    for i in range(num_images):
      data = [item[0][i].repeat(3, 1, 1) if item[0][i].size(0) == 1 else item[0][i][:3] for item in batch]  # Extract data (list of tensors)
      #print(len(data))
      
      data = torch.stack(data)
      outer_list.append(data)
    
    data = tuple(outer_list)
    labels = [item[1] for item in batch]  # Extract labels (list of integers)
    
    # Convert labels to a tensor
    stacked_labels = torch.tensor(labels)
    
    return data, stacked_labels