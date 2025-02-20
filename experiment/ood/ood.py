import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image
import faiss
import numpy as np
from experiment.utils.get_num_workers import get_num_workers


class OOD:
    def __init__(
        self,
        args,
        dataset,
        feature_extractor,
        cycle_idx=None,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):
        self.dataset = dataset
        self.num_workers = min(6, get_num_workers())
        self.feature_extractor = feature_extractor
        self.batch_size = args.val_batch_size
        self.K = args.k
        self.num_ood_samples = args.num_ood_samples
        self.cycle_idx = cycle_idx
        self.device = device
        self.dtype = dtype

    def extract_features(self):
        """Extract and normalize features from the entire dataset"""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep order for proper index mapping
            num_workers=self.num_workers,
            pin_memory=True,
        )

        features = []
        indices = []

        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):
                # Keep track of original indices
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + len(batch)
                indices.extend(range(start_idx, end_idx))

                # Extract features
                batch = batch.to(device=self.device, dtype=self.dtype)
                batch_features = self.feature_extractor(batch)

                if batch_features.dim() == 1:
                    batch_features = batch_features.unsqueeze(0)

                # Normalize and move to CPU
                batch_features = F.normalize(batch_features, p=2, dim=-1)
                features.append(batch_features.cpu())

        features = torch.cat(features, dim=0)

        # Verify normalization
        norms = torch.norm(features, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-3):
            print("Warning: Features not properly normalized")

        return features.numpy().astype(np.float32), indices

    def compute_outlier_scores(self, features):
        """Compute outlier scores using average distance to k nearest neighbors"""
        print("\nComputing outlier scores...")
        n_samples = len(features)

        # Build FAISS index
        dimension = features.shape[1]
        index = faiss.IndexFlatL2(dimension)

        # Add normalized features to index
        index.add(features)

        # For each point, find its k+1 nearest neighbors (including self)
        k = min(self.K + 1, n_samples)
        distances, indices = index.search(features, k)

        # Compute outlier scores based on distance to k-nearest neighbors
        # Exclude self-distances (first column) and take max distance instead of mean
        outlier_scores = np.max(distances[:, 1:], axis=1)

        # Print some statistics
        print(f"\nOutlier score statistics:")
        print(f"Mean: {np.mean(outlier_scores):.4f}")
        print(f"Std: {np.std(outlier_scores):.4f}")
        print(f"Min: {np.min(outlier_scores):.4f}")
        print(f"Max: {np.max(outlier_scores):.4f}")

        return outlier_scores

    def ood(self):
        """Identify most out-of-distribution samples"""
        # Extract and normalize features
        features, indices = self.extract_features()
        print(f"\nFeature matrix shape: {features.shape}")

        # Compute outlier scores
        outlier_scores = self.compute_outlier_scores(features)

        # Determine number of samples to select
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get indices of highest scoring samples
        top_indices = np.argsort(outlier_scores)[-num_samples:][::-1]

        # Print top outliers
        print("\nTop outlier scores:")
        for idx in top_indices:
            print(f"Index {indices[idx]}: score {outlier_scores[idx]:.4f}")

        # Map to original dataset indices
        ood_indices = [indices[i] for i in top_indices]

        # Save results
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        np.save(f"./ood_logs/{self.cycle_idx}/outlier_scores.npy", outlier_scores)

        # Save example images
        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, _ = self.dataset[ood_indices[i]]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                score = outlier_scores[top_indices[i]]
                image_path = (
                    f"./ood_logs/{self.cycle_idx}/images/ood_{i}_score_{score:.3f}.jpg"
                )
                save_image(image, image_path)

        return ood_indices
