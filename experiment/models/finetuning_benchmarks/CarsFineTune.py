import os
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
import scipy.io
import PIL.Image as Image
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import os
import pandas as pd
from PIL import Image
import warnings
from experiment.utils.get_num_workers import get_num_workers

from .TransferLearningBenchmark import TransferLearningBenchmark


class StanfordCars(Dataset):
    """
    Stanford Cars dataset
    """

    def __init__(self, root, split="train", transform=None, download=False):
        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self.transform = transform

        self.data_dir = os.path.join(self.root, self.__class__.__name__)

        # New URLs from Stanford's CS231N course mirrors
        self.urls = {
            "train_images": "http://cs231n.stanford.edu/car_data/cars_train.tgz",
            "test_images": "http://cs231n.stanford.edu/car_data/cars_test.tgz",
            "devkit": "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            "test_annotations": "http://cs231n.stanford.edu/car_data/cars_test_annos_withlabels.mat",
        }

        if download:
            self._download()

        self._load_metadata()

    def _download(self):
        if os.path.exists(self.data_dir):
            return

        os.makedirs(self.data_dir, exist_ok=True)

        # Download and extract all files
        for key, url in self.urls.items():
            filename = os.path.basename(url)
            download_and_extract_archive(
                url, self.data_dir, filename=filename, remove_finished=True
            )

    def _load_metadata(self):
        # Load class names
        meta_path = os.path.join(self.data_dir, "devkit", "cars_meta.mat")
        meta = scipy.io.loadmat(meta_path)
        self.classes = [str(label[0]) for label in meta["class_names"][0]]

        # Load annotations
        if self.split == "train":
            annot_path = os.path.join(self.data_dir, "devkit", "cars_train_annos.mat")
            img_dir = os.path.join(self.data_dir, "cars_train")
        else:
            annot_path = os.path.join(self.data_dir, "cars_test_annos_withlabels.mat")
            img_dir = os.path.join(self.data_dir, "cars_test")

        annos = scipy.io.loadmat(annot_path)

        # Format: [bbox_x1, bbox_y1, bbox_x2, bbox_y2, class]
        self.samples = []
        annotations = (
            annos["annotations"][0] if self.split == "train" else annos["annotations"]
        )

        for anno in annotations:
            img_name = anno[-1][0] if self.split == "train" else f"{anno[5][0]}.jpg"
            class_id = anno[-2][0][0] if self.split == "train" else anno[-1][0][0]
            img_path = os.path.join(img_dir, img_name)
            # Class IDs are 1-indexed in the dataset
            self.samples.append((img_path, class_id - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CarsFineTune(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=196, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_train_dataset = StanfordCars(
                root="data", split="train", download=True, transform=self.transform
            )
            test_dataset = StanfordCars(
                root="data", split="test", download=True, transform=self.transform
            )

        # Split train into train/val
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        return train_dataset, val_dataset, test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
