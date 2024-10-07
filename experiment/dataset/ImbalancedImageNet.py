import os
from typing import TypedDict
from PIL import Image
from random import random
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from datasets import load_dataset
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)


class DataPoint(TypedDict):
    image: Image.Image
    label: int


class DummyImageNet(Dataset):
    def __init__(self, num_classes: int, transform=None):
        super().__init__()

        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return 100

    def __getitem__(self, _) -> tuple[Image.Image, int]:
        datapoint: DataPoint = {"image": Image.new("RGB", (224, 224)), "label": 0}

        if self.transform:
            datapoint["image"] = self.transform(datapoint["image"])

        return datapoint["image"], datapoint["label"]


class ImbalancedImageNet(Dataset):
    def __init__(
        self,
        dataset_path: str,
        imbalance_method: ImbalanceMethod = ImbalanceMethods.LinearlyIncreasing,
        checkpoint_filename: str = None,
        transform=None,
    ):
        super().__init__()

        self.checkpoint_filename = checkpoint_filename
        self.transform = transform
        split = "train+validation"

        self.dataset = load_dataset(dataset_path, split=split)

        self.classes = self.dataset.features["label"].names
        self.num_classes = len(self.classes)
        self.imbalancedness = imbalance_method.value.impl(self.num_classes)
        self.indices = self._load_or_create_indices()
        self.additional_data = self._create_or_load_additional_data()

        print("original length:", len(self.dataset))
        print("imbalanced length:", len(self))

    def save_additional_datapoint(self, filename: str, label: int):
        self.additional_data.append((filename, label))
        self._save_additional_data_to_pickle()

    def _create_or_load_additional_data(self):
        try:
            return self._load_additional_data_from_pickle()
        except FileNotFoundError:
            return []

    @property
    def _additional_data_filename(self):
        dataset_pickles = os.environ["BASE_CACHE_DIR"] + "/dataset_pickles"

        if not os.path.exists(dataset_pickles):
            os.makedirs(dataset_pickles)

        return f"{dataset_pickles}/{self.checkpoint_filename}_additional_data.pkl"

    def _save_additional_data_to_pickle(self):
        with open(self._additional_data_filename, "wb") as f:
            pickle.dump(self.additional_data, f)

    def _load_additional_data_from_pickle(self):
        with open(self._additional_data_filename, "rb") as f:
            return pickle.load(f)

    def _create_indices(self) -> list[int]:
        """
        The imbalancedness class assigns an imbalance score
        between 0 and 1 to each class. Based on this, we randomly
        add a fraction of *1 - imbalance* of the samples of each class
        to our dataset.
        """
        indices = []

        for index, sample in tqdm(
            enumerate(self.dataset),
            total=len(self.dataset),
            desc="Making dataset imbalanced",
        ):
            if self.imbalancedness.get_imbalance(sample["label"]) < random():
                indices.append(index)

        self._save_indices_to_pickle(indices)

        return indices

    @property
    def _indices_filename(self):
        dataset_pickles = os.environ["BASE_CACHE_DIR"] + "/dataset_pickles"

        if not os.path.exists(dataset_pickles):
            os.makedirs(dataset_pickles)

        return f"{dataset_pickles}/{self.checkpoint_filename}_indices.pkl"

    def _save_indices_to_pickle(self, indices: list[int]):
        with open(self._indices_filename, "wb") as f:
            pickle.dump(indices, f)

    def _load_indices_from_pickle(self):
        with open(self._indices_filename, "rb") as f:
            return pickle.load(f)

    def _load_or_create_indices(self):
        return self._create_indices()

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

        return {"image": Image.open(filename), "label": label}

    def __getitem__(self, idx) -> DataPoint:
        """
        Our indices vector maps from each index in our dataset
        to an index in the original dataset.
        We return the sample at the index in the original dataset.
        """
        datapoint = (
            self.dataset[self.indices[idx]]
            if idx < len(self.indices)
            else self._load_additional_datapoint(idx - len(self.indices))
        )

        datapoint["image"] = datapoint["image"].convert("RGB")

        if self.transform:
            datapoint["image"] = self.transform(datapoint["image"])

        return datapoint["image"], datapoint["label"]
