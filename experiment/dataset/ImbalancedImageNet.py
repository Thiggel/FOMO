from typing import TypedDict
from PIL import Image
from random import random
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import pickle

from datasets import load_dataset
from experiment.dataset.imbalancedness.ImbalanceMethods import \
    ImbalanceMethods, ImbalanceMethod


class DataPoint(TypedDict):
    image: Image.Image
    label: int


class ImbalancedImageNet(Dataset):
    def __init__(
        self,
        dataset_path: str,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        transform=None
    ):
        super().__init__()

        self.train_dataset = load_dataset(dataset_path, split='train')
        self.val_dataset = load_dataset(dataset_path, split='validation')

        self.dataset = ConcatDataset([
            self.train_dataset,
            self.val_dataset
        ])

        self.transform = transform
        self.classes = self.train_dataset.features['label'].names
        self.num_classes = len(self.classes)
        self.imbalancedness = imbalance_method.value.impl(len(self.classes))
        self.indices = self._load_or_create_indices()
        self.checkpoint_filename = checkpoint_filename
        self.additional_data = self._create_or_load_additional_data()

    def _save_additional_datapoint(self, filename: str, label: int):
        self.additional_data.append((filename, label))
        self._save_additional_data_to_pickle()

    def _create_or_load_additional_data(self):
        try:
            return self._load_additional_data_from_pickle()
        except FileNotFoundError:
            return []

    def _save_additional_data_to_pickle(self):
        with open('additional_data.pkl', 'wb') as f:
            pickle.dump(self.additional_data, f)

    def _load_additional_data_from_pickle(self):
        with open('additional_data.pkl', 'rb') as f:
            return pickle.load(f)

    def _create_indices(self) -> list[int]:
        """
        The imbalancedness class assigns an imbalance score
        between 0 and 1 to each class. Based on this, we randomly
        add a fraction of *1 - imbalance* of the samples of each class
        to our dataset.
        """
        indices = []

        for index, sample in tqdm(enumerate(self.dataset), total=len(self.dataset), desc='Making dataset imbalanced'):
            if self.imbalancedness.get_imbalance(sample['label']) < random():
                indices.append(index)

        self._save_indices_to_pickle(indices)

        return indices

    def _save_indices_to_pickle(self, indices: list[int]):
        with open('indices.pkl', 'wb') as f:
            pickle.dump(indices, f)

    def _load_indices_from_pickle(self):
        with open('indices.pkl', 'rb') as f:
            return pickle.load(f)

    def _load_or_create_indices(self):
        try:
            return self._load_indices_from_pickle()
        except FileNotFoundError:
            return self._create_indices()

        return self.indices

    def __len__(self):
        """
        The number of samples in the dataset is the number of indices
        which is smaller than the original dataset due to the imbalance.
        """
        return len(self.indices) + len(self.additional_data)

    def _load_additional_datapoint(self, idx) -> DataPoint:
        """
        We return the sample at the index in the additional data.
        """
        filename, label = self.additional_data[idx]

        return {
            'image': Image.open(filename),
            'label': label
        }

    def __getitem__(self, idx) -> DataPoint:
        """
        Our indices vector maps from each index in our dataset
        to an index in the original dataset.
        We return the sample at the index in the original dataset.
        """
        datapoint = self.dataset[self.indices[idx]] \
            if idx < len(self.indices) \
            else self._load_additional_datapoint(idx - len(self.indices))

        if self.transform:
            datapoint['image'] = self.transform(datapoint['image'])

        return datapoint['image'], datapoint['label']