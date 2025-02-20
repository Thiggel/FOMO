import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import shutil
from pathlib import Path
import numpy as np
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


class TestImbalancedImageNet(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_storage")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BASE_CACHE_DIR"] = str(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_basic_functionality(self):
        """Test basic dataset initialization and access"""
        dataset = ImbalancedImageNet(
            dataset_path="sxdave/emotion_detection",
            additional_data_path="test_additional_data",
            imbalance_method=ImbalanceMethods.LinearlyIncreasing,
            checkpoint_filename="test_checkpoint",
        )

        # Check initial state
        initial_len = len(dataset)
        print(f"\nInitial dataset length: {initial_len}")
        print(f"Initial indices length: {len(dataset.indices)}")

        # Try accessing an item
        image, label = dataset[0]
        self.assertIsNotNone(image)
        self.assertIsNotNone(label)

    def test_adding_generated_images(self):
        """Test adding and accessing generated images"""
        dataset = ImbalancedImageNet(
            dataset_path="sxdave/emotion_detection",
            additional_data_path="test_additional_data",
            imbalance_method=ImbalanceMethods.LinearlyIncreasing,
            checkpoint_filename="test_checkpoint",
        )

        # Record initial state
        initial_len = len(dataset)
        print(f"\nInitial length: {initial_len}")

        # Add some generated images
        cycle_idx = 0
        num_images = 3
        new_images = [Image.new("RGB", (24, 24)) for _ in range(num_images)]

        # Save the images
        dataset.image_storage.save_batch(new_images, cycle_idx=cycle_idx, start_idx=0)
        dataset.add_generated_images(cycle_idx, num_images, [0] * num_images)

        print(f"Length after adding images: {len(dataset)}")
        print(f"Additional image counts: {dataset.additional_image_counts}")

        self.assertEqual(len(dataset), initial_len + num_images)

        # Try accessing the new images
        for i in range(num_images):
            idx = initial_len + i
            try:
                image, label = dataset[idx]
                print(f"Successfully accessed image at index {idx}")
            except Exception as e:
                print(f"Error accessing image at index {idx}: {str(e)}")
                raise

    def test_dataloader_compatibility(self):
        """Test if the dataset works with DataLoader"""
        dataset = ImbalancedImageNet(
            dataset_path="sxdave/emotion_detection",
            additional_data_path="test_additional_data",
            imbalance_method=ImbalanceMethods.LinearlyIncreasing,
            checkpoint_filename="test_checkpoint",
        )

        # Add some generated images
        cycle_idx = 0
        num_images = 3
        new_images = [Image.new("RGB", (24, 24)) for _ in range(num_images)]
        dataset.image_storage.save_batch(new_images, cycle_idx=cycle_idx, start_idx=0)
        dataset.add_generated_images(cycle_idx, num_images, [0] * num_images)

        # Try using DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        try:
            for batch_idx, (images, labels) in enumerate(loader):
                print(f"Successfully loaded batch {batch_idx}")
                self.assertIsNotNone(images)
                self.assertIsNotNone(labels)
        except Exception as e:
            print(f"Error in DataLoader: {str(e)}")
            raise


if __name__ == "__main__":
    unittest.main()
