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
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        features = []
        indices = torch.arange(len(self.dataset))

        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):

                # Extract features
                batch = batch.to(device=self.device, dtype=self.dtype)
                batch_features = self.feature_extractor(batch)

                # Handle single-dimension case
                if batch_features.dim() == 1:
                    batch_features = batch_features.unsqueeze(0)

                # Normalize features
                batch_features = F.normalize(batch_features, p=2, dim=-1)

                # Move to CPU immediately to free GPU memory
                features.append(batch_features.cpu())

        # Concatenate all features
        features = torch.cat(features, dim=0).numpy().astype("float32")
        return features, indices

    def compute_knn_distances(self, features):
        """Compute k-NN distances using CPU FAISS"""
        print("\nBuilding FAISS index...")
        dimension = features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(features)

        # Calculate k (number of neighbors to find)
        k = min(self.K + 1, len(features))  # +1 because we'll remove self-distance

        print("\nSearching for nearest neighbors...")
        distances, _ = index.search(features, k)

        # Remove self-distances (first column) and compute mean
        knn_distances = distances[:, 1:].mean(axis=1)

        return knn_distances

    def ood(self):
        """Perform OOD detection and return indices of most OOD samples"""
        # Extract features
        features, indices = self.extract_features()

        # Compute KNN distances
        knn_distances = self.compute_knn_distances(features)

        # Calculate number of samples if percentage given
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get top-K most distant samples
        top_indices = np.argsort(knn_distances)[-num_samples:]

        # Map back to original dataset indices
        ood_indices = [indices[i] for i in top_indices]

        # Save logs and visualizations
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        # Save distances for analysis
        np.save(f"./ood_logs/{self.cycle_idx}/knn_distances.npy", knn_distances)

        # Save example OOD images
        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, _ = self.dataset[ood_indices[i]]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                score = knn_distances[top_indices[-(i + 1)]]
                image_path = (
                    f"./ood_logs/{self.cycle_idx}/images/ood_{i}_score_{score:.3f}.jpg"
                )
                save_image(image, image_path)

        return ood_indices
