import os
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
from torchvision import transforms
import warnings
from PIL import Image
import scipy.io as sio
from .TransferLearningBenchmark import TransferLearningBenchmark

class StanfordCars(Dataset):
    def __init__(self, root_path, annot_path=None, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        # Get all image files and sort them to ensure consistent ordering
        self.images = sorted([f for f in os.listdir(root_path) if f.endswith('.jpg')])
        
        # Load annotations if provided
        self.labels = None
        if annot_path and os.path.exists(annot_path):
            annotations = sio.loadmat(annot_path)
            if 'annotations' in annotations:
                # The class labels in the dataset are 1-indexed
                self.labels = [int(anno[4][0][0] - 1) for anno in annotations['annotations'][0]]
                # Sort by filename to match image order
                filename_label_pairs = [(os.path.basename(anno[0][0]), anno[4][0][0] - 1) 
                                     for anno in annotations['annotations'][0]]
                filename_label_pairs.sort(key=lambda x: x[0])
                self.labels = [label for _, label in filename_label_pairs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = os.path.join(self.root_path, self.images[index])
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Return both image and label if labels are available, otherwise just image
        if self.labels is not None:
            return image, self.labels[index]
        return image, -1  # Return -1 as label if no annotations available

class CarsFineTune(TransferLearningBenchmark):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        **kwargs
    ):
        super().__init__(
            model=model, lr=lr, transform=transform, num_classes=196, *args, **kwargs
        )
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        base_path = os.getenv("BASE_CACHE_DIR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_dataset = StanfordCars(
                root_path=os.path.join(base_path, 'cars_train/cars_train'),
                annot_path=os.path.join(base_path, 'cars_train_annos.mat'),
                transform=self.transform
            )
            test_dataset = StanfordCars(
                root_path=os.path.join(base_path, 'cars_test/cars_test'),
                annot_path=os.path.join(base_path, 'cars_test_annos_withlabels.mat'),
                transform=self.transform
            )

        # Split train into train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
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
