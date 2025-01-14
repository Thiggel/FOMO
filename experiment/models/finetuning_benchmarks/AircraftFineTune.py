import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import os
from PIL import Image, UnidentifiedImageError
import warnings

from .TransferLearningBenchmark import TransferLearningBenchmark


class FGVCAircraft(Dataset):
    """
    FGVC Aircraft dataset
    """

    def __init__(self, root, split="train", transform=None, download=False):
        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.transform = transform

        self.url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        self.class_types = "variant"  # Use variant level annotations (100 classes)

        if download:
            self._download()

        self._load_metadata()

    def _download(self):
        if os.path.exists(os.path.join(self.root, "fgvc-aircraft-2013b")):
            print(os.listdir(os.path.join(self.root)))
            return

        download_and_extract_archive(
            self.url,
            self.root,
            extract_root=self.root,
            filename="fgvc-aircraft-2013b.tar.gz",
            remove_finished=True,
        )

    def _load_metadata(self):
        import scipy.io

        data_dir = os.path.join(self.root, "fgvc-aircraft-2013b")

        # Load variant names
        variants_path = os.path.join(data_dir, "data", "variants.txt")
        with open(variants_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Load image paths and labels based on split
        split_map = {
            "train": "images_variant_train.txt",
            "val": "images_variant_val.txt",
            "test": "images_variant_test.txt",
        }

        annotations_path = os.path.join(data_dir, "data", split_map[self.split])
        with open(annotations_path, "r") as f:
            content = [line.strip().split(" ", 1) for line in f.readlines()]
            self.samples = [
                (
                    os.path.join(data_dir, "data", "images", f"{img}.jpg"),
                    self.class_to_idx[label],
                )
                for img, label in content
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path, target = self.samples[idx]
            img = Image.open(path).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
        except UnidentifiedImageError:
            print(f"Error loading image at index {idx}. Skipping...")
            return self.__getitem__(idx + 1)

        return img, target


class AircraftFineTune(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=100, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        base_path = os.getenv("BASE_CACHE_DIR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_dataset = FGVCAircraft(
                root=base_path + "/data", split="train", download=True, transform=self.transform
            )
            val_dataset = FGVCAircraft(
                root=base_path + "/data", split="val", download=True, transform=self.transform
            )
            test_dataset = FGVCAircraft(
                root=base_path + "/data", split="test", download=True, transform=self.transform
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
