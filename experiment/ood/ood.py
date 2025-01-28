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
        train,
        test,
        feature_extractor,
        cycle_idx=None,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):
        self.train = train
        self.test = test
        self.num_workers = min(24, get_num_workers())
        self.feature_extractor = feature_extractor
        self.batch_size = args.fe_batch_size
        self.K = args.k
        self.pct_ood = args.pct_ood
        self.cycle_idx = cycle_idx
        self.device = device
        self.dtype = dtype

        # Will store feature batches
        self.train_features_batches = []
        self.test_features_batches = []

    @torch.cuda.amp.autocast()
    def extract_features(self):
        train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # Extract and store features in batches
        with torch.no_grad():
            for batch, _ in tqdm(train_loader, desc="Extracting train features"):
                batch = batch.to(device=self.device, dtype=self.dtype)
                features = self.feature_extractor(batch)
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                self.train_features_batches.append(features)

            for batch, _ in tqdm(test_loader, desc="Extracting test features"):
                batch = batch.to(device=self.device, dtype=self.dtype)
                features = self.feature_extractor(batch)
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                self.test_features_batches.append(features)

    def compute_distances(self, query_batch):
        distances_per_batch = []

        # Normalize query batch
        query_batch = F.normalize(query_batch, p=2, dim=-1)

        # Compare with each train batch
        for train_batch in self.train_features_batches:
            train_batch = F.normalize(train_batch, p=2, dim=-1)
            # Compute similarity for current batch pair
            similarity = torch.matmul(query_batch, train_batch.T)
            # Convert to distance
            dist = 2 - 2 * similarity
            distances_per_batch.append(dist)

        # Concatenate along the correct dimension to get all distances for this query batch
        return torch.cat(distances_per_batch, dim=1)

    def ood(self):
        all_scores = []

        # Process each test batch
        for test_batch in tqdm(self.test_features_batches, desc="Computing OOD scores"):
            # Get distances from this test batch to all training samples
            distances = self.compute_distances(test_batch)

            # Get top-K smallest distances for each sample in batch
            k = min(
                self.K, sum(batch.shape[0] for batch in self.train_features_batches)
            )
            topk_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)

            # Use k-th nearest neighbor distance as anomaly score
            batch_scores = topk_distances[:, -1]
            all_scores.append(batch_scores)

        # Combine scores from all batches
        scores_ood = torch.cat(all_scores).float()

        # Compute threshold and OOD indices
        threshold = torch.quantile(scores_ood, 1 - self.pct_ood)
        is_ood = scores_ood >= threshold
        ood_indices = torch.where(is_ood)[0]

        # Save results
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}", exist_ok=True)

        torch.save(scores_ood, f"./ood_logs/{self.cycle_idx}/scores_ood.pt")

        # Save top-K most OOD images
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}/images"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images")

        top_k_indices = torch.argsort(scores_ood, descending=True)[:10]
        for i, idx in enumerate(top_k_indices):
            image = self.test[int(idx)][0]
            if len(image.shape) in [3, 4]:
                distance = scores_ood[idx].item()
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}.jpg"
                save_image(image, image_path)

        return ood_indices, threshold
