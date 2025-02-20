import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import shutil
import numpy as np
from pathlib import Path
import h5py
import io

from experiment.dataset.ImageStorage import ImageStorage
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet, DummyImageNet


class TestImageStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.storage = ImageStorage(base_path, max_images_per_file=10)

    def cleanup(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)


class DummyImbalancedDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.data = [
            (Image.new("RGB", (24, 24)), i % num_classes) for i in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class TestDatasetAddition(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_storage")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.storage = TestImageStorage(str(self.test_dir))

    def tearDown(self):
        self.storage.cleanup()

    def test_basic_image_storage(self):
        """Test basic image storage and retrieval"""
        # Create a test image
        test_image = Image.new("RGB", (24, 24), color="red")

        # Save the image
        self.storage.storage.save_image(test_image, cycle_idx=0, global_idx=0)

        # Retrieve the image
        retrieved_image = self.storage.storage.load_image(cycle_idx=0, global_idx=0)

        self.assertIsNotNone(retrieved_image)
        self.assertEqual(retrieved_image.size, (24, 24))

    def test_batch_storage(self):
        """Test storing and retrieving multiple images in a batch"""
        batch_size = 5
        images = [Image.new("RGB", (24, 24), color="blue") for _ in range(batch_size)]

        # Save batch
        self.storage.storage.save_batch(images, cycle_idx=0, start_idx=0)

        # Retrieve each image
        for i in range(batch_size):
            retrieved = self.storage.storage.load_image(cycle_idx=0, global_idx=i)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.size, (24, 24))

    def test_multiple_cycles(self):
        """Test storing images across multiple cycles"""
        cycles = [0, 1, 2]
        images_per_cycle = 3

        for cycle in cycles:
            images = [Image.new("RGB", (24, 24)) for _ in range(images_per_cycle)]
            self.storage.storage.save_batch(images, cycle_idx=cycle, start_idx=0)

        # Verify images from each cycle
        for cycle in cycles:
            for idx in range(images_per_cycle):
                retrieved = self.storage.storage.load_image(
                    cycle_idx=cycle, global_idx=idx
                )
                self.assertIsNotNone(retrieved)

    def test_clear_cycle(self):
        """Test clearing images from a specific cycle"""
        # Save some images
        images = [Image.new("RGB", (24, 24)) for _ in range(3)]
        self.storage.storage.save_batch(images, cycle_idx=0, start_idx=0)

        # Clear the cycle
        self.storage.storage.clear_cycle(0)

        # Verify images are gone
        retrieved = self.storage.storage.load_image(cycle_idx=0, global_idx=0)
        self.assertIsNone(retrieved)


class TestImbalancedImageNet(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_storage")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BASE_CACHE_DIR"] = str(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dataset_growth(self):
        """Test that the dataset properly grows when adding generated images"""
        # Create a dummy dataset
        dataset = DummyImageNet(num_classes=10)

        # Initialize storage
        storage = ImageStorage(str(self.test_dir))

        # Save some generated images
        cycle_idx = 0
        num_new_images = 5
        new_images = [Image.new("RGB", (24, 24)) for _ in range(num_new_images)]
        storage.save_batch(new_images, cycle_idx=cycle_idx, start_idx=0)

        # Add the generated images to the dataset
        initial_len = len(dataset)
        dataset.add_generated_images(cycle_idx, num_new_images, [0] * num_new_images)

        self.assertEqual(len(dataset), initial_len + num_new_images)

    def test_multiple_cycles_addition(self):
        """Test adding images across multiple cycles"""
        dataset = DummyImageNet(num_classes=10)
        storage = ImageStorage(str(self.test_dir))

        cycles = [0, 1]
        images_per_cycle = 3
        initial_len = len(dataset)

        for cycle in cycles:
            new_images = [Image.new("RGB", (24, 24)) for _ in range(images_per_cycle)]
            storage.save_batch(new_images, cycle_idx=cycle, start_idx=0)
            dataset.add_generated_images(
                cycle, images_per_cycle, [0] * images_per_cycle
            )

        expected_len = initial_len + (len(cycles) * images_per_cycle)
        self.assertEqual(len(dataset), expected_len)

    def test_data_loading(self):
        """Test that added images can be properly loaded"""
        dataset = DummyImageNet(num_classes=10)
        storage = ImageStorage(str(self.test_dir))

        # Add some images
        cycle_idx = 0
        num_images = 5
        new_images = [
            Image.new("RGB", (24, 24), color="red") for _ in range(num_images)
        ]
        storage.save_batch(new_images, cycle_idx=cycle_idx, start_idx=0)
        dataset.add_generated_images(cycle_idx, num_images, [0] * num_images)

        # Create a DataLoader
        loader = DataLoader(dataset, batch_size=2)

        # Try loading the data
        for batch in loader:
            images, labels = batch
            self.assertIsNotNone(images)
            self.assertIsNotNone(labels)
            self.assertEqual(images.dim(), 4)  # B x C x H x W

    def test_label_consistency(self):
        """Test that labels are preserved when adding generated images"""
        dataset = DummyImageNet(num_classes=10)
        storage = ImageStorage(str(self.test_dir))

        # Add images with specific labels
        cycle_idx = 0
        num_images = 3
        labels = [1, 2, 3]  # Specific labels for testing
        new_images = [Image.new("RGB", (24, 24)) for _ in range(num_images)]
        storage.save_batch(new_images, cycle_idx=cycle_idx, start_idx=0)
        dataset.add_generated_images(cycle_idx, num_images, labels)

        # Verify labels
        for i, label in enumerate(labels):
            _, retrieved_label = dataset[len(dataset) - num_images + i]
            self.assertEqual(retrieved_label, label)


if __name__ == "__main__":
    unittest.main()
