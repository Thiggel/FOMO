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
        """Compute k-NN distances using FAISS GPU"""
        features_np = self.features.cpu().numpy().astype("float32")

        # Create FAISS index
        d = features_np.shape[1]  # dimension
        res = faiss.StandardGpuResources()  # GPU resource object
        index = faiss.GpuIndexFlatL2(res, d)  # GPU index

        # Add vectors to index
        index.add(features_np)

        # Search for k+1 nearest neighbors (including self)
        k = min(self.K + 1, len(features_np))
        distances, _ = index.search(features_np, k)

        # Remove self-distances (first column) and compute mean of remaining distances
        knn_distances = distances[:, 1:].mean(axis=1)

        return torch.from_numpy(knn_distances).to(self.device), self.indices

    def ood(self):
        """Perform OOD detection and return N most OOD samples based on k-NN distances"""
        self.extract_features()
        knn_scores, original_indices = self.compute_knn_distances()

        # Calculate number of samples if percentage given
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get indices of N samples with largest k-NN distances
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
