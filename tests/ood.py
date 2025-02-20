import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import shutil
import faiss
import gc
import os


class DummyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_features(self, x):
        return x


class Args:
    def __init__(self):
        self.val_batch_size = 32
        self.k = 3
        self.num_ood_samples = 2


def cleanup_memory():
    """Helper function to clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestOOD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test if FAISS is working properly
        try:
            d = 10  # dimension
            nb = 100  # database size
            xb = np.random.random((nb, d)).astype("float32")
            index = faiss.IndexFlatL2(d)
            index.add(xb)
            print("FAISS test successful")
        except Exception as e:
            raise unittest.SkipTest(f"FAISS not working properly: {str(e)}")

    def setUp(self):
        cleanup_memory()
        # Import OOD here to ensure experiment module is in path
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from experiment.ood.ood import OOD

        self.OOD = OOD
        self.args = Args()

        # Create test directory
        self.test_dir = Path("./ood_logs/test")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        cleanup_memory()
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_tiny_dataset(self):
        """Test with a very small dataset to verify basic functionality"""
        try:
            # Create a tiny dataset with clear outliers
            normal = torch.tensor(
                [[1.0, 1.0], [1.2, 1.1], [0.8, 0.9]], dtype=torch.float32
            )
            outlier = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

            features = torch.cat([normal, outlier])
            labels = torch.zeros(len(features))

            dataset = DummyDataset(features, labels)
            model = DummyModel()

            # Request 1 OOD sample
            self.args.num_ood_samples = 1
            self.args.k = 2  # Use k=2 for tiny dataset

            ood = self.OOD(
                args=self.args,
                dataset=dataset,
                feature_extractor=model.extract_features,
                cycle_idx="test",
                device=torch.device("cpu"),
            )

            ood_indices = ood.ood()

            # The outlier should be detected
            self.assertEqual(len(ood_indices), 1)
            self.assertEqual(ood_indices[0], 3)  # Last index should be the outlier

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_small_clusters(self):
        """Test with small clusters to avoid memory issues"""
        try:
            # Create two small clusters and one outlier
            cluster1 = torch.randn(5, 2) * 0.1
            cluster2 = torch.randn(5, 2) * 0.1 + torch.tensor([2.0, 2.0])
            outlier = torch.tensor([[10.0, 10.0]])

            features = torch.cat([cluster1, cluster2, outlier])
            labels = torch.zeros(len(features))

            dataset = DummyDataset(features, labels)
            model = DummyModel()

            self.args.num_ood_samples = 1
            self.args.k = 3

            ood = self.OOD(
                args=self.args,
                dataset=dataset,
                feature_extractor=model.extract_features,
                cycle_idx="test",
                device=torch.device("cpu"),
            )

            ood_indices = ood.ood()

            # Verify that the outlier is detected
            self.assertEqual(len(ood_indices), 1)
            self.assertEqual(ood_indices[0], len(features) - 1)

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_normalized_input(self):
        """Test with normalized input vectors"""
        try:
            # Create small normalized vectors
            features = torch.randn(10, 3)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            # Add one clearly different but normalized vector
            outlier = torch.ones(1, 3) / np.sqrt(3)
            features = torch.cat([features, outlier])

            labels = torch.zeros(len(features))

            dataset = DummyDataset(features, labels)
            model = DummyModel()

            self.args.num_ood_samples = 1
            self.args.k = 3

            ood = self.OOD(
                args=self.args,
                dataset=dataset,
                feature_extractor=model.extract_features,
                cycle_idx="test",
                device=torch.device("cpu"),
            )

            ood_indices = ood.ood()

            # Verify that the outlier is detected
            self.assertEqual(len(ood_indices), 1)
            self.assertEqual(ood_indices[0], len(features) - 1)

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_k_values(self):
        """Test different k values"""
        try:
            # Create a simple dataset
            features = torch.randn(20, 2)
            outlier = torch.tensor([[100.0, 100.0]])
            features = torch.cat([features, outlier])
            labels = torch.zeros(len(features))

            dataset = DummyDataset(features, labels)
            model = DummyModel()

            # Test with different k values
            k_values = [1, 3, 5]
            for k in k_values:
                self.args.k = k
                self.args.num_ood_samples = 1

                ood = self.OOD(
                    args=self.args,
                    dataset=dataset,
                    feature_extractor=model.extract_features,
                    cycle_idx="test",
                    device=torch.device("cpu"),
                )

                ood_indices = ood.ood()

                # The outlier should always be detected regardless of k
                self.assertEqual(ood_indices[0], len(features) - 1)

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
