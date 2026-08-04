import os
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, annotations_file=None, transform=None, test=False):
        self.root_dir = os.path.realpath(os.path.expanduser(root_dir))
        self.transform = transform

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Stanford Cars directory does not exist: {self.root_dir}"
            )

        self.image_paths = [
            os.path.join(self.root_dir, filename)
            for filename in os.listdir(self.root_dir)
            if filename.endswith(".jpg")
        ]

        if annotations_file:
            self.annotations_file = sio.loadmat(os.path.expanduser(annotations_file))
            self.annotations = self.annotations_file["annotations"][
                0
            ]  # Load annotations
            if test:
                self.filename_to_label = {
                    ann[4][0]: int(ann[0][0][0]) for ann in self.annotations
                }  # Assign -1 to all test images
            else:
                self.filename_to_label = {
                    ann[5][0]: int(ann[4][0][0]) for ann in self.annotations
                }  # Create mapping
        else:
            self.filename_to_label = (
                {}
            )  # Empty dictionary if no annotations file is provided

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format

        # Get label if available
        filename = os.path.basename(image_path)
        label = self.filename_to_label.get(
            filename, -1
        )  # Use -1 as default label if not in the annotations

        if label == -1:
            print("No label found for image: ", filename, ". Skipping...")
            return self.__getitem__(idx + 1)

        if self.transform:
            image = self.transform(image)

        # Always return a label even if -1
        return image, label - 1


class HuggingFaceStanfordCarsDataset(Dataset):
    """Stanford Cars using a maintained parquet mirror.

    Torchvision no longer downloads Stanford Cars because the original
    Stanford host is unavailable. This mirror retains the official 8,144/8,041
    train/test split and avoids relying on a manually reconstructed devkit.
    """

    dataset_id = "tanganke/stanford_cars"

    def __init__(self, split: str, transform=None):
        self.dataset = load_dataset(self.dataset_id, split=split)
        self.transform = transform
        self.classes = self.dataset.features["label"].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[int(idx)]
        image = sample["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])
