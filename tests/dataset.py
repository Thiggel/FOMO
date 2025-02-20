import unittest
import torch
from torch.utils.data import DataLoader
from PIL import Image
import os
import shutil
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.dataset.ImageStorage import ImageStorage


class DummyData:
    """Minimal implementation matching the HuggingFace dataset interface"""

    def __init__(self, size, num_classes):
        self.size = size
        self.data = [(torch.randn(3, 224, 224), i % num_classes) for i in range(size)]
        self.features = {"label": DummyFeature(num_classes)}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[i] for i in idx]
        tensor, label = self.data[idx]
        # Convert tensor to PIL Image for compatibility
        image = transforms.ToPILImage()(tensor)
        return {"image": image, "label": label}


class DummyFeature:
    def __init__(self, num_classes):
        self.names = [str(i) for i in range(num_classes)]


class MockImbalancedImageNet(ImbalancedImageNet):
    """Modified ImbalancedImageNet that uses a dummy dataset"""

    def __init__(self, test_dir, *args, **kwargs):
        self.test_dir = test_dir
        os.environ["BASE_CACHE_DIR"] = str(test_dir)

        # Initialize with dummy values
        super().__init__(
            dataset_path="dummy",
            additional_data_path=str(test_dir / "additional_data"),
            imbalance_method=ImbalanceMethods.LinearlyIncreasing,
            checkpoint_filename="test_checkpoint",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        # Override dataset with dummy data
        self.num_classes = 10
        self.dataset = DummyData(100, self.num_classes)
        self.classes = [str(i) for i in range(self.num_classes)]

        # Create initial indices
        self.indices = list(
            range(0, 50)
        )  # Use half the data to simulate imbalanced dataset

        # Reinitialize storage with test path
        self.image_storage = ImageStorage(
            str(test_dir / "additional_data"), max_images_per_file=10
        )

        # Initialize additional image counts
        self.additional_image_counts = {}
        self._save_image_counts()


class TestImbalancedImageNet(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_storage")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset
        self.dataset = MockImbalancedImageNet(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initial_state(self):
        """Test initial dataset state"""
        self.assertEqual(self.dataset.num_classes, 10)
        self.assertEqual(len(self.dataset.indices), 50)  # Initial imbalanced size
        self.assertEqual(len(self.dataset), 50)  # Should match indices length initially

    def test_add_generated_images(self):
        """Test adding generated images to the dataset"""
        initial_len = len(self.dataset)

        # Generate some random images
        cycle_idx = 0
        num_images = 5
        new_images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(num_images)
        ]

        # Save images
        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )

        # Add to dataset
        labels = [0] * num_images
        self.dataset.add_generated_images(cycle_idx, num_images, labels)

        # Verify length increased
        self.assertEqual(len(self.dataset), initial_len + num_images)

        # Verify we can access the new images
        for i in range(num_images):
            idx = initial_len + i
            image, label = self.dataset[idx]
            self.assertIsNotNone(image)
            self.assertEqual(label, 0)

    def test_multiple_cycles(self):
        """Test adding images across multiple cycles"""
        initial_len = len(self.dataset)
        images_per_cycle = 3
        cycles = [0, 1]

        for cycle in cycles:
            new_images = [
                Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                for _ in range(images_per_cycle)
            ]
            self.dataset.image_storage.save_batch(
                new_images, cycle_idx=cycle, start_idx=0
            )
            self.dataset.add_generated_images(
                cycle, images_per_cycle, [0] * images_per_cycle
            )

        expected_len = initial_len + (len(cycles) * images_per_cycle)
        self.assertEqual(len(self.dataset), expected_len)

    def test_data_loading(self):
        """Test loading data through DataLoader"""
        # Add some generated images first
        cycle_idx = 0
        num_images = 5
        new_images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(num_images)
        ]
        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )
        self.dataset.add_generated_images(cycle_idx, num_images, [0] * num_images)

        # Create DataLoader
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        # Try loading batches
        num_batches = 0
        for images, labels in loader:
            self.assertEqual(images.dim(), 4)
            self.assertEqual(images.size(1), 3)
            self.assertEqual(images.size(2), 224)
            self.assertEqual(images.size(3), 224)
            self.assertEqual(labels.dim(), 1)
            num_batches += 1

        # Verify we got the expected number of batches
        expected_batches = (len(self.dataset) + 1) // 2  # Ceiling division
        self.assertEqual(num_batches, expected_batches)

    def test_generated_images_persistence(self):
        """Test that generated images persist across dataset reloads"""
        # Add some generated images
        cycle_idx = 0
        num_images = 3
        new_images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(num_images)
        ]
        labels = [1, 2, 3]

        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )
        self.dataset.add_generated_images(cycle_idx, num_images, labels)

        # Record the length
        original_len = len(self.dataset)

        # Create a new dataset instance
        new_dataset = MockImbalancedImageNet(self.test_dir)

        # Verify lengths match
        self.assertEqual(
            len(new_dataset),
            original_len,
            f"Expected length {original_len}, got {len(new_dataset)}",
        )

        # Verify we can access the generated images with correct labels
        base_len = len(new_dataset.indices)
        for i in range(num_images):
            idx = base_len + i
            image, label = new_dataset[idx]
            self.assertIsNotNone(image)
            self.assertEqual(label, labels[i])


if __name__ == "__main__":
    unittest.main()
