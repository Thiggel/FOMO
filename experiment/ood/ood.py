from typing import Optional

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
        self.num_ood_samples = args.num_ood_samples * args.every_nth_ood_sample
        self.every_nth_ood_sample = args.every_nth_ood_sample
        self.cycle_idx = cycle_idx
        self.device = device
        self.dtype = dtype
        self.selection_strategy = getattr(args, "ood_selection_strategy", "top")
        self.mode_histogram_bins = getattr(args, "ood_mode_histogram_bins", "auto")
        self.last_results: Optional[dict] = None

    def extract_features(self):
        """Extract features from the dataset without normalization"""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        features = []
        indices = []

        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + len(batch)
                indices.extend(range(start_idx, end_idx))

                # Extract features
                batch = batch.to(device=self.device, dtype=self.dtype)
                batch_features = self.feature_extractor(batch)

                if batch_features.dim() == 1:
                    batch_features = batch_features.unsqueeze(0)

                # Do NOT normalize features - we want to preserve actual distances
                features.append(batch_features.cpu())

        # Concatenate all features
        features = torch.cat(features, dim=0)

        print(f"\nFeature statistics:")
        print(f"Mean: {features.mean():.4f}")
        print(f"Std: {features.std():.4f}")
        print(f"Min: {features.min():.4f}")
        print(f"Max: {features.max():.4f}")

        torch.cuda.empty_cache()

        return features.numpy().astype(np.float32), indices

    def compute_knn_distances(self, features):
        """Compute mean k-NN distance for each point"""
        print("\nComputing KNN distances...")
        n_samples = len(features)

        # Create FAISS index
        dimension = features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(features)

        # Find k+1 nearest neighbors (including self)
        k = min(self.K + 1, n_samples)
        distances, neighbors = index.search(features, k)

        # Remove self-distance (first column) and compute mean
        knn_distances = distances[:, 1:].mean(axis=1)

        print(f"\nDistance statistics:")
        print(f"Mean distance: {np.mean(knn_distances):.4f}")
        print(f"Std distance: {np.std(knn_distances):.4f}")
        print(f"Min distance: {np.min(knn_distances):.4f}")
        print(f"Max distance: {np.max(knn_distances):.4f}")

        return knn_distances

    def ood(self):
        """Identify the most out-of-distribution samples"""
        # Extract features (without normalization)
        features, indices = self.extract_features()

        # Compute distances
        distances = self.compute_knn_distances(features)

        # Calculate number of samples to select
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get indices ordered by distance
        sorted_indices_desc = np.argsort(distances)[::-1]

        if self.selection_strategy == "mode_window":
            selected_indices, mode_details = self._select_mode_window_indices(
                distances, num_samples
            )
        else:
            selected_indices = sorted_indices_desc[:num_samples]
            mode_details = None

        selected_indices = np.array(selected_indices, dtype=int)

        # Map to original dataset indices
        selected_dataset_indices = [indices[i] for i in selected_indices]

        # Apply sub-sampling if requested
        selected_indices_step = selected_indices[:: self.every_nth_ood_sample]
        selected_dataset_indices_step = selected_dataset_indices[
            :: self.every_nth_ood_sample
        ]

        # Store details for downstream analysis/logging
        self.last_results = {
            "distances": distances,
            "dataset_indices": indices,
            "sorted_indices": sorted_indices_desc,
            "selected_indices": selected_indices,
            "selected_dataset_indices": selected_dataset_indices,
            "selected_indices_step": selected_indices_step,
            "selected_dataset_indices_step": selected_dataset_indices_step,
            "top_indices": selected_indices,
            "top_dataset_indices": selected_dataset_indices,
            "selection_strategy": self.selection_strategy,
        }

        if mode_details is not None:
            self.last_results["mode_details"] = mode_details

        # Save results and visualizations
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        np.save(f"./ood_logs/{self.cycle_idx}/distances.npy", distances)

        num_vis = min(10, len(selected_dataset_indices_step))
        for i in range(num_vis):
            dataset_idx = selected_dataset_indices_step[i]
            image, _ = self.dataset[dataset_idx]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                distance = distances[selected_indices_step[i]]
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}.jpg"
                save_image(image, image_path)

        # select only every nth ood sample
        ood_indices = selected_dataset_indices_step

        return ood_indices

    def _select_mode_window_indices(self, distances, num_samples):
        """Select indices around the mode of the distance distribution."""
        if num_samples <= 0:
            return np.array([], dtype=int), None

        total_samples = len(distances)
        if total_samples == 0:
            return np.array([], dtype=int), None

        num_samples = min(num_samples, total_samples)

        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        try:
            hist_density, bin_edges = np.histogram(
                sorted_distances, bins=self.mode_histogram_bins, density=True
            )
        except TypeError:
            hist_density, bin_edges = np.histogram(
                sorted_distances, bins="auto", density=True
            )

        if hist_density.size == 0:
            return sorted_indices[:num_samples], None

        mode_bin_idx = int(np.argmax(hist_density))
        mode_left = float(bin_edges[mode_bin_idx])
        mode_right = float(bin_edges[mode_bin_idx + 1])
        mode_center = (mode_left + mode_right) / 2

        lower_target = num_samples // 2
        upper_target = num_samples - lower_target

        lower_positions = np.where(sorted_distances < mode_center)[0]
        upper_positions = np.where(sorted_distances >= mode_center)[0]

        selected_positions = []

        if lower_target > 0 and lower_positions.size > 0:
            take_lower = min(lower_target, lower_positions.size)
            selected_positions.extend(lower_positions[-take_lower:].tolist())

        if upper_target > 0 and upper_positions.size > 0:
            take_upper = min(upper_target, upper_positions.size)
            selected_positions.extend(upper_positions[:take_upper].tolist())

        selected_positions = np.array(selected_positions, dtype=int)

        remaining = num_samples - selected_positions.size
        if remaining > 0:
            all_positions = np.arange(total_samples, dtype=int)
            mask = np.ones(total_samples, dtype=bool)
            if selected_positions.size > 0:
                mask[selected_positions] = False
            candidate_positions = all_positions[mask]
            if candidate_positions.size > 0:
                order = np.argsort(
                    np.abs(sorted_distances[candidate_positions] - mode_center)
                )
                extra_positions = candidate_positions[order[:remaining]]
                selected_positions = (
                    np.concatenate([selected_positions, extra_positions])
                    if selected_positions.size > 0
                    else extra_positions
                )

        # Ensure uniqueness and exact count by prioritizing proximity to the mode
        if selected_positions.size == 0:
            selected_positions = np.array([np.argmin(np.abs(sorted_distances - mode_center))])

        selected_positions = np.unique(selected_positions)
        if selected_positions.size > num_samples:
            order = np.argsort(
                np.abs(sorted_distances[selected_positions] - mode_center)
            )
            selected_positions = selected_positions[order[:num_samples]]

        if selected_positions.size < num_samples:
            all_positions = np.arange(total_samples, dtype=int)
            mask = np.ones(total_samples, dtype=bool)
            mask[selected_positions] = False
            candidate_positions = all_positions[mask]
            if candidate_positions.size > 0:
                order = np.argsort(
                    np.abs(sorted_distances[candidate_positions] - mode_center)
                )
                needed = min(num_samples - selected_positions.size, candidate_positions.size)
                selected_positions = np.concatenate(
                    [selected_positions, candidate_positions[order[:needed]]]
                )

        order = np.argsort(
            np.abs(sorted_distances[selected_positions] - mode_center)
        )
        selected_positions = selected_positions[order[:num_samples]]

        mode_details = {
            "mode_center": mode_center,
            "mode_bin_start": mode_left,
            "mode_bin_end": mode_right,
            "hist_density": hist_density.tolist(),
            "hist_bin_edges": bin_edges.tolist(),
        }

        return sorted_indices[selected_positions], mode_details
