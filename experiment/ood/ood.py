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

        self.features = []
        self.indices = []

    @torch.cuda.amp.autocast()
    def extract_features(self):
        """Extract and normalize features from the entire dataset"""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + len(batch)
                self.indices.extend(range(start_idx, end_idx))

                batch = batch.to(device=self.device, dtype=self.dtype)
                features = self.feature_extractor(batch)

                if features.dim() == 1:
                    features = features.unsqueeze(0)

                features = F.normalize(features, p=2, dim=-1)
                self.features.append(features)

        self.features = torch.cat(self.features)
        self.indices = torch.tensor(self.indices, device=self.device)

    def compute_knn_distances(self):
        """Compute average distance to k nearest neighbors efficiently on GPU"""
        num_samples = self.features.size(0)
        k = min(self.K, num_samples - 1)

        all_knn_scores = []
        all_indices = []
        batch_size = 512

        for i in tqdm(
            range(0, num_samples, batch_size), desc="Computing k-NN distances"
        ):
            end_idx = min(i + batch_size, num_samples)
            query_features = self.features[i:end_idx]

            # Compute pairwise distances
            distances = 2 - 2 * torch.matmul(query_features, self.features.t())

            # Mask out self-distances
            mask = torch.arange(i, end_idx, device=self.device).unsqueeze(
                1
            ) == torch.arange(num_samples, device=self.device)
            distances.masked_fill_(mask, float("inf"))

            # Get k-nearest neighbors
            knn_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)

            # Compute average distance to k nearest neighbors
            knn_scores = knn_distances.mean(dim=1)

            # Store scores and indices
            all_knn_scores.append(knn_scores)
            all_indices.append(self.indices[i:end_idx])

        return torch.cat(all_knn_scores), torch.cat(all_indices)

    def ood(self):
        """Perform OOD detection and return N most OOD samples based on average k-NN distance"""
        self.extract_features()
        knn_scores, original_indices = self.compute_knn_distances()

        # Calculate number of samples if percentage given
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get indices of N samples with largest average k-NN distances
        top_scores, top_idx = torch.topk(knn_scores, k=num_samples, largest=True)
        ood_indices = original_indices[top_idx].cpu()

        # Save visualizations and logs
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        torch.save(knn_scores, f"./ood_logs/{self.cycle_idx}/knn_scores.pt")

        # Save example OOD images
        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, _ = self.dataset[int(ood_indices[i])]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                score = top_scores[i]
                image_path = (
                    f"./ood_logs/{self.cycle_idx}/images/ood_{i}_score_{score:.3f}.jpg"
                )
                save_image(image, image_path)

        return ood_indices.tolist()
