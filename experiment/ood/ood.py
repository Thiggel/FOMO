import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image

from experiment.utils.get_num_workers import get_num_workers


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image
from experiment.utils.get_num_workers import get_num_workers


class OOD:
    def __init__(
        self,
        args,
        dataset,  # Single dataset instead of train/test split
        feature_extractor,
        cycle_idx=None,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):
        self.dataset = dataset
        self.num_workers = min(24, get_num_workers())
        self.feature_extractor = feature_extractor
        self.batch_size = args.val_batch_size
        self.K = args.k  # For ImageNet100, we use k=100
        self.num_ood_samples = args.num_ood_samples  # Hyperparameter for threshold
        self.cycle_idx = cycle_idx
        self.device = device
        self.dtype = dtype

        print("DATASET", dataset)

        # Will store features and corresponding indices
        self.features = []
        self.indices = []  # Store original dataset indices

    @torch.cuda.amp.autocast()
    def extract_features(self):
        """Extract and normalize features from the entire dataset"""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Important: keep order for index tracking
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # Extract and normalize features using GPU
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):
                # Track indices for this batch
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + len(batch)
                self.indices.extend(range(start_idx, end_idx))

                # Extract features
                batch = batch.to(device=self.device, dtype=self.dtype)
                features = self.feature_extractor(batch)

                # Handle 1D case
                if features.dim() == 1:
                    features = features.unsqueeze(0)

                # L2 normalize features as emphasized in paper
                features = F.normalize(features, p=2, dim=-1)
                self.features.append(features)

        # Concatenate all features and convert indices to tensor
        self.features = torch.cat(self.features)
        self.indices = torch.tensor(self.indices, device=self.device)

    def compute_knn_distances(self):
        """Compute k-NN distances efficiently on GPU using batched operations"""
        num_samples = self.features.size(0)
        k = min(self.K, num_samples - 1)  # Ensure k is valid

        all_distances = []
        batch_size = 512  # Process in batches to manage memory

        for i in tqdm(
            range(0, num_samples, batch_size), desc="Computing k-NN distances"
        ):
            # Get batch of query features
            end_idx = min(i + batch_size, num_samples)
            query_features = self.features[i:end_idx]

            # Compute pairwise L2 distances with all features
            # Using ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            # Since features are normalized, ||x||^2 = ||y||^2 = 1
            distances = 2 - 2 * torch.matmul(query_features, self.features.t())

            # Mask out self-distances
            mask = torch.arange(i, end_idx, device=self.device).unsqueeze(
                1
            ) == torch.arange(num_samples, device=self.device)
            distances.masked_fill_(mask, float("inf"))

            # Get k-nearest neighbor distances
            knn_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)
            all_distances.append(
                knn_distances[:, -1]
            )  # Keep only k-th nearest neighbor distance

        return torch.cat(all_distances)

    def ood(self):
        """Perform OOD detection and return N most OOD samples"""
        # Extract normalized features
        self.extract_features()

        # Compute k-NN distances
        distances = self.compute_knn_distances()

        # Get indices of N samples with largest k-NN distances
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )
        top_distances, top_indices = torch.topk(distances, k=num_samples, largest=True)

        # Map back to original dataset indices
        ood_indices = self.indices[top_indices].cpu()

        # Save visualizations and logs
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        # Save distances distribution
        torch.save(distances, f"./ood_logs/{self.cycle_idx}/distances.pt")

        # Save example OOD images (save top 10 most OOD samples)
        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, _ = self.dataset[int(ood_indices[i])]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                distance = top_distances[i]
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}.jpg"
                save_image(image, image_path)

        return ood_indices.tolist()
