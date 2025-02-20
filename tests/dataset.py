import unittest
import torch
from torch.utils.data import DataLoader
from PIL import Image
import os
import shutil
from pathlib import Path
import torchvision.transforms as transforms
from experiment.dataset.ImbalancedImageNet import ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.dataset.ImageStorage import ImageStorage


class MockImbalancedImageNet(ImbalancedImageNet):
    """Modified ImbalancedImageNet that uses random tensors instead of real images"""

    def __init__(self, *args, **kwargs):
        # Override dataset loading with a simple dummy dataset
        super().__init__(
            dataset_path="sxdave/emotion_detection",  # This won't actually be loaded
            additional_data_path="test_additional_data",
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

        # Initialize storage with test path
        self.image_storage = ImageStorage(
            self.additional_data_path, max_images_per_file=10
        )


class DummyData:
    """Simple dummy dataset for testing"""

    def __init__(self, size, num_classes):
        self.size = size
        self.data = [(torch.randn(3, 224, 224), i % num_classes) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tensor, label = self.data[idx]
        # Convert tensor to PIL Image for compatibility
        image = transforms.ToPILImage()(tensor)
        return {"image": image, "label": label}


class TestImbalancedImageNet(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_storage")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BASE_CACHE_DIR"] = str(self.test_dir)

        # Initialize dataset
        self.dataset = MockImbalancedImageNet()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("test_additional_data"):
            shutil.rmtree("test_additional_data")

    def test_initial_state(self):
        """Test initial dataset state"""
        self.assertEqual(self.dataset.num_classes, 10)
        self.assertGreater(len(self.dataset), 0)

    def test_add_generated_images(self):
        """Test adding generated images to the dataset"""
        initial_len = len(self.dataset)

        # Generate some random images
        cycle_idx = 0
        num_images = 5
        new_images = [
            Image.fromarray((torch.randn(224, 224, 3) * 255).byte().numpy())
            for _ in range(num_images)
        ]

        # Save images
        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )

        # Add to dataset
        labels = [0] * num_images  # Using same label for simplicity
        self.dataset.add_generated_images(cycle_idx, num_images, labels)

        # Verify length increased
        self.assertEqual(len(self.dataset), initial_len + num_images)

    def test_multiple_cycles(self):
        """Test adding images across multiple cycles"""
        initial_len = len(self.dataset)
        images_per_cycle = 3
        cycles = [0, 1]

        for cycle in cycles:
            # Generate and add images for each cycle
            new_images = [
                Image.fromarray((torch.randn(224, 224, 3) * 255).byte().numpy())
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
        # Add some generated images
        cycle_idx = 0
        num_images = 5
        new_images = [
            Image.fromarray((torch.randn(224, 224, 3) * 255).byte().numpy())
            for _ in range(num_images)
        ]
        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )
        self.dataset.add_generated_images(cycle_idx, num_images, [0] * num_images)

        # Create DataLoader
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        # Try loading batches
        for images, labels in loader:
            self.assertEqual(images.dim(), 4)  # B x C x H x W
            self.assertEqual(images.size(0), min(2, len(self.dataset)))  # Batch size
            self.assertEqual(images.size(1), 3)  # RGB channels
            self.assertEqual(images.size(2), 224)  # Height
            self.assertEqual(images.size(3), 224)  # Width
            self.assertEqual(labels.dim(), 1)  # 1D tensor of labels

    def test_access_generated_images(self):
        """Test accessing generated images after adding them"""
        # Add some generated images
        cycle_idx = 0
        num_images = 3
        new_images = [
            Image.fromarray((torch.randn(224, 224, 3) * 255).byte().numpy())
            for _ in range(num_images)
        ]
        labels = [1, 2, 3]  # Different labels for testing

        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )
        self.dataset.add_generated_images(cycle_idx, num_images, labels)

        # Access the newly added images
        initial_data_len = len(self.dataset.indices)
        for i in range(num_images):
            idx = initial_data_len + i
            image, label = self.dataset[idx]

            self.assertIsInstance(
                image, torch.Tensor
            )  # Should be tensor after transform
            self.assertEqual(image.size(), (3, 224, 224))  # Check dimensions
            self.assertEqual(label, labels[i])  # Check label

    def test_generated_images_persistence(self):
        """Test that generated images persist across dataset reloads"""
        # Add some generated images
        cycle_idx = 0
        num_images = 3
        new_images = [
            Image.fromarray((torch.randn(224, 224, 3) * 255).byte().numpy())
            for _ in range(num_images)
        ]
        labels = [1, 2, 3]

        self.dataset.image_storage.save_batch(
            new_images, cycle_idx=cycle_idx, start_idx=0
        )
        self.dataset.add_generated_images(cycle_idx, num_images, labels)

        # Create a new dataset instance
        new_dataset = MockImbalancedImageNet()

        # Verify the images are still accessible
        self.assertEqual(len(new_dataset), len(self.dataset))

        initial_data_len = len(new_dataset.indices)
        for i in range(num_images):
            idx = initial_data_len + i
            image, label = new_dataset[idx]
            self.assertEqual(label, labels[i])


if __name__ == "__main__":
    unittest.main()
