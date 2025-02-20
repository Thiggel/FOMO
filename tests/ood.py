def test_tiny_dataset(self):
    """Test with a very small dataset to verify basic functionality"""
    try:
        # Create a tiny dataset with clear outliers
        normal = torch.tensor([[1.0, 1.0], [1.2, 1.1], [0.8, 0.9]], dtype=torch.float32)
        outlier = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

        features = torch.cat([normal, outlier])
        labels = torch.zeros(len(features))

        # Print dataset info
        print("\nDataset features:")
        for i, feat in enumerate(features):
            print(f"Index {i}: {feat}")

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

        # Print distances for all points
        print("\nVerifying detected outlier:")
        print(f"Selected index: {ood_indices[0]}")
        print(f"Selected point: {features[ood_indices[0]]}")
        print(f"Expected outlier: {features[-1]}")

        # The outlier should be detected
        self.assertEqual(len(ood_indices), 1)
        self.assertEqual(
            ood_indices[0],
            len(features) - 1,
            f"Expected last index ({len(features)-1}) but got {ood_indices[0]}",
        )

    except Exception as e:
        self.fail(f"Test failed with error: {str(e)}")


def test_small_clusters(self):
    """Test with small clusters to avoid memory issues"""
    try:
        # Create two small clusters and one outlier
        np.random.seed(42)  # For reproducibility
        cluster1 = torch.randn(5, 2) * 0.1
        cluster2 = torch.randn(5, 2) * 0.1 + torch.tensor([2.0, 2.0])
        outlier = torch.tensor([[10.0, 10.0]])

        features = torch.cat([cluster1, cluster2, outlier])
        labels = torch.zeros(len(features))

        # Print dataset info
        print("\nDataset structure:")
        print(f"Cluster 1: indices 0-4")
        print(f"Cluster 2: indices 5-9")
        print(f"Outlier: index {len(features)-1}")

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

        # Print debug info
        print("\nSelected point info:")
        print(f"Selected index: {ood_indices[0]}")
        print(f"Selected point coordinates: {features[ood_indices[0]]}")
        print(f"Outlier coordinates: {features[-1]}")

        # Verify that the outlier is detected
        self.assertEqual(len(ood_indices), 1)
        self.assertEqual(
            ood_indices[0],
            len(features) - 1,
            f"Expected index {len(features)-1}, got {ood_indices[0]}",
        )

    except Exception as e:
        self.fail(f"Test failed with error: {str(e)}")


def test_normalized_input(self):
    """Test with normalized input vectors"""
    try:
        # Create small normalized vectors
        torch.manual_seed(42)  # For reproducibility
        features = torch.randn(10, 3)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Add one clearly different but normalized vector
        outlier = torch.ones(1, 3) / np.sqrt(3)
        features = torch.cat([features, outlier])

        # Print dataset info
        print("\nFeature vectors:")
        for i, feat in enumerate(features):
            norm = torch.norm(feat)
            print(f"Index {i}: norm={norm:.6f}, vector={feat}")

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

        # Print debug info
        print("\nSelected point info:")
        print(f"Selected index: {ood_indices[0]}")
        print(f"Selected point: {features[ood_indices[0]]}")
        print(f"Expected outlier: {features[-1]}")

        # Verify that the outlier is detected
        self.assertEqual(len(ood_indices), 1)
        self.assertEqual(
            ood_indices[0],
            len(features) - 1,
            f"Expected last index ({len(features)-1}) but got {ood_indices[0]}",
        )

    except Exception as e:
        self.fail(f"Test failed with error: {str(e)}")


def test_k_values(self):
    """Test different k values"""
    try:
        # Create a simple dataset
        torch.manual_seed(42)  # For reproducibility
        features = torch.randn(20, 2)
        outlier = torch.tensor([[100.0, 100.0]])
        features = torch.cat([features, outlier])

        # Print dataset info
        print("\nDataset info:")
        print(f"Normal points: indices 0-19")
        print(f"Outlier: index 20")
        print(f"Outlier coordinates: {outlier[0]}")

        labels = torch.zeros(len(features))
        dataset = DummyDataset(features, labels)
        model = DummyModel()

        # Test with different k values
        k_values = [1, 3, 5]
        for k in k_values:
            print(f"\nTesting with k={k}")
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

            # Print debug info
            print(f"Selected index: {ood_indices[0]}")
            print(f"Selected point: {features[ood_indices[0]]}")

            # The outlier should always be detected regardless of k
            self.assertEqual(
                ood_indices[0],
                len(features) - 1,
                f"k={k}: Expected last index ({len(features)-1}) but got {ood_indices[0]}",
            )

    except Exception as e:
        self.fail(f"Test failed with error: {str(e)}")
