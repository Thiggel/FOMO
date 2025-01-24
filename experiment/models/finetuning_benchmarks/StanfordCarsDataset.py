import os
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, annotations_file=None, transform=None, test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, filename)
            for filename in os.listdir(root_dir)
            if filename.endswith(".jpg")
        ]

        print(annotations_file)
        if annotations_file:
            self.annotations_file = sio.loadmat(annotations_file)
            print(self.annotations_file.keys())
            print(self.annotations_file["__globals__"])
            self.annotations = self.annotations_file["annotations"][
                0
            ]  # Load annotations
            print(self.annotations)
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

        if self.transform:
            image = self.transform(image)

        # Always return a label even if -1
        return image, label
