from typing import Optional
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
        # Keep the full configuration because a few acquisition variants use
        # optional controls that are not otherwise materialized as attributes.
        self.args = args
        self.dataset = dataset
        self.num_workers = min(6, get_num_workers())
        self.feature_extractor = feature_extractor
        self.batch_size = args.val_batch_size
        self.K = args.k
        self.distance_metric = str(
            getattr(args, "ood_distance_metric", "normalized_l2")
        ).lower()
        self.num_ood_samples = args.num_ood_samples * args.every_nth_ood_sample
        self.every_nth_ood_sample = args.every_nth_ood_sample
        self.cycle_idx = cycle_idx
        self.device = device
        self.dtype = dtype
        self.selection_strategy = getattr(args, "ood_selection_strategy", "top")
        self.mode_histogram_bins = getattr(args, "ood_mode_histogram_bins", "auto")
        self.mode_histogram_quantile_range = self._parse_quantile_range(
            getattr(args, "ood_mode_histogram_quantile_range", (0.01, 0.99))
        )
        self.mode_candidate_pool_multiplier = max(
            1.0, float(getattr(args, "ood_mode_candidate_pool_multiplier", 1.0))
        )
        self.mode_diversity_sampling = bool(
            getattr(args, "ood_mode_diversity_sampling", False)
        )
        self.mode_diversity_normalize_features = bool(
            getattr(args, "ood_mode_diversity_normalize_features", True)
        )
        # Keep diagnostics with each experiment run.  Writing to a shared
        # repository-level ood_logs directory corrupts evidence when Slurm
        # array tasks run concurrently.
        run_dir = getattr(args, "additional_data_path", None)
        self.diagnostics_dir = (
            os.path.join(str(run_dir), "ood_diagnostics")
            if run_dir
            else "./ood_logs"
        )
        self.last_results: Optional[dict] = None
        self.mode_histogram_max_bins = 512

    @staticmethod
    def _parse_quantile_range(value):
        if isinstance(value, str):
            cleaned = value.strip().strip("[]()")
            parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(
                    "ood_mode_histogram_quantile_range must contain exactly two values"
                )
            low, high = (float(parts[0]), float(parts[1]))
        else:
            low, high = (float(value[0]), float(value[1]))

        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                "ood_mode_histogram_quantile_range must satisfy 0 <= low < high <= 1"
            )

        return (low, high)

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
            for batch_idx, loaded_batch in enumerate(tqdm(loader, desc="Extracting features")):
                batch = loaded_batch[0]
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

        index_features = np.asarray(features, dtype=np.float32)
        if self.distance_metric in {
            "normalized_l2",
            "cosine",
            "median_normalized",
        }:
            norms = np.linalg.norm(index_features, axis=1, keepdims=True)
            index_features = index_features / np.clip(norms, 1e-12, None)
        elif self.distance_metric != "raw_l2":
            raise ValueError(
                "ood_distance_metric must be one of raw_l2, normalized_l2, "
                "cosine, or median_normalized"
            )

        # Create FAISS index
        dimension = index_features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(index_features)

        # Find k+1 nearest neighbors (including self)
        k = min(self.K + 1, n_samples)
        distances, neighbors = index.search(index_features, k)

        # Remove self-distance (first column) and compute mean
        knn_distances = distances[:, 1:].mean(axis=1)
        if self.distance_metric == "cosine":
            # For unit vectors, squared Euclidean distance is 2(1-cosine).
            knn_distances = knn_distances / 2.0
        elif self.distance_metric == "median_normalized":
            median = float(np.median(knn_distances))
            knn_distances = knn_distances / max(median, 1e-12)

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

        percentile_bin = getattr(self.args, "ood_percentile_bin", None)
        if percentile_bin is not None:
            low_q, high_q = [float(value) for value in percentile_bin]
            low, high = np.quantile(distances, [low_q, high_q])
            candidates = np.flatnonzero((distances >= low) & (distances <= high))
            selected_indices = self._fps_dataset_indices(
                candidates, features, num_samples
            )
            mode_details = {
                "percentile_bin": [low_q, high_q],
                "candidate_pool_size": int(len(candidates)),
                "diversity_sampling": True,
            }
        elif self.selection_strategy == "dense":
            selected_indices = np.argsort(distances)[:num_samples]
            mode_details = {"dense_region_placebo": True}
        elif self.selection_strategy in {"mode_window", "mode_random"}:
            diversity = self.mode_diversity_sampling
            if self.selection_strategy == "mode_random":
                self.mode_diversity_sampling = False
            selected_indices, mode_details = self._select_mode_window_indices(
                distances, num_samples, features
            )
            self.mode_diversity_sampling = diversity
        elif self.selection_strategy == "densest_window":
            selected_indices, mode_details = self._select_densest_window_indices(
                distances, num_samples, features
            )
        elif self.selection_strategy in {"band_random", "band_fps"}:
            selected_indices, mode_details = self._select_sparse_band_indices(
                distances,
                num_samples,
                features,
                use_fps=self.selection_strategy == "band_fps",
            )
        elif self.selection_strategy == "all_fps":
            positions = np.arange(len(distances), dtype=int)
            selected_indices = self._fps_dataset_indices(
                positions, features, num_samples
            )
            mode_details = {"candidate_pool_size": int(len(positions))}
        elif self.selection_strategy == "cluster_inverse":
            selected_indices, mode_details = self._select_inverse_cluster_indices(
                features, num_samples
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

        # Persist the actual policy output alongside the score distribution.
        # This is needed for the rebuttal's score-percentile, overlap, and
        # original-versus-synthetic provenance analyses; a histogram alone
        # cannot establish what the selector actually acquired.
        sorted_ascending = np.sort(distances)
        selected_scores = distances[selected_indices]
        selected_percentiles = np.searchsorted(
            sorted_ascending, selected_scores, side="right"
        ) / max(1, len(sorted_ascending))
        selected_underlying, selected_original = self._selection_provenance(
            selected_dataset_indices
        )
        self.last_results.update(
            {
                "selected_scores": selected_scores,
                "selected_score_percentiles": selected_percentiles,
                "selected_underlying_indices": selected_underlying,
                "selected_is_original": selected_original,
            }
        )

        # Save results and visualizations
        cycle_dir = os.path.join(self.diagnostics_dir, str(self.cycle_idx))
        if not os.path.exists(cycle_dir):
            os.makedirs(os.path.join(cycle_dir, "images"), exist_ok=True)

        np.save(os.path.join(cycle_dir, "distances.npy"), distances)
        np.savez_compressed(
            os.path.join(cycle_dir, "selection.npz"),
            dataset_indices=np.asarray(indices, dtype=np.int64),
            distances=np.asarray(distances, dtype=np.float32),
            selected_positions=np.asarray(selected_indices, dtype=np.int64),
            selected_dataset_indices=np.asarray(selected_dataset_indices, dtype=np.int64),
            selected_underlying_indices=np.asarray(selected_underlying, dtype=np.int64),
            selected_scores=np.asarray(selected_scores, dtype=np.float32),
            selected_score_percentiles=np.asarray(selected_percentiles, dtype=np.float32),
            selected_is_original=np.asarray(selected_original, dtype=bool),
        )
        summary = {
            "selection_strategy": str(self.selection_strategy),
            "distance_metric": str(self.distance_metric),
            "k": int(self.K),
            "n_candidates": int(len(distances)),
            "n_selected": int(len(selected_indices)),
            "selected_percentile_median": float(np.median(selected_percentiles)),
            "selected_percentile_min": float(np.min(selected_percentiles)),
            "selected_percentile_max": float(np.max(selected_percentiles)),
            "selected_original_fraction": float(np.mean(selected_original)),
            "mode_details": mode_details,
        }
        with open(os.path.join(cycle_dir, "selection.json"), "w") as handle:
            json.dump(summary, handle, indent=2)

        num_vis = min(10, len(selected_dataset_indices_step))
        for i in range(num_vis):
            dataset_idx = selected_dataset_indices_step[i]
            image = self.dataset[dataset_idx][0]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                distance = distances[selected_indices_step[i]]
                image_path = os.path.join(
                    cycle_dir, "images", f"ood_{i}_distance_{distance:.3f}.jpg"
                )
                save_image(image, image_path)

        # select only every nth ood sample
        ood_indices = selected_dataset_indices_step

        return ood_indices

    def _selection_provenance(self, selected_dataset_indices):
        """Resolve Subset nesting and mark anchors as source or synthetic."""
        underlying = []
        root = self.dataset
        while isinstance(root, Subset):
            root = root.dataset
        original_pool_size = (
            len(root.indices) if hasattr(root, "indices") else None
        )

        for position in selected_dataset_indices:
            dataset = self.dataset
            index = int(position)
            while isinstance(dataset, Subset):
                if not 0 <= index < len(dataset.indices):
                    raise IndexError(
                        f"Selection position {index} is outside a Subset of "
                        f"length {len(dataset.indices)}"
                    )
                index = int(dataset.indices[index])
                dataset = dataset.dataset
            underlying.append(index)

        if original_pool_size is None:
            original = [True] * len(underlying)
        else:
            original = [index < original_pool_size for index in underlying]
        return underlying, original

    def _select_mode_window_indices(self, distances, num_samples, features=None):
        """Select indices around the mode of the distance distribution."""
        if num_samples <= 0:
            return np.array([], dtype=int), None

        total_samples = len(distances)
        if total_samples == 0:
            return np.array([], dtype=int), None

        num_samples = min(num_samples, total_samples)

        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        finite_mask = np.isfinite(sorted_distances)

        if not finite_mask.all():
            sorted_indices = sorted_indices[finite_mask]
            sorted_distances = sorted_distances[finite_mask]
            total_samples = len(sorted_distances)
            if total_samples == 0:
                return np.array([], dtype=int), None
            num_samples = min(num_samples, total_samples)

        if np.isclose(sorted_distances[0], sorted_distances[-1]):
            clipped_low = float(sorted_distances[0])
            clipped_high = float(sorted_distances[-1])
            mode_left = float(sorted_distances[0])
            mode_right = float(sorted_distances[-1])
            mode_center = mode_left
            hist_density = np.array([float(total_samples)], dtype=np.float64)
            bin_edges = np.array([mode_left - 0.5, mode_right + 0.5], dtype=np.float64)
        else:
            quantile_low, quantile_high = self.mode_histogram_quantile_range
            clipped_low, clipped_high = np.quantile(
                sorted_distances, [quantile_low, quantile_high]
            )

            if not np.isfinite(clipped_low) or not np.isfinite(clipped_high):
                clipped_low = float(sorted_distances[0])
                clipped_high = float(sorted_distances[-1])

            if clipped_high <= clipped_low:
                clipped_low = float(sorted_distances[0])
                clipped_high = float(sorted_distances[-1])

            histogram_values = sorted_distances[
                (sorted_distances >= clipped_low) & (sorted_distances <= clipped_high)
            ]
            if histogram_values.size < 2 or np.isclose(clipped_low, clipped_high):
                histogram_values = sorted_distances
                clipped_low = float(sorted_distances[0])
                clipped_high = float(sorted_distances[-1])

            bins = self.mode_histogram_bins
            if isinstance(bins, str) and bins.isdigit():
                bins = int(bins)

            if isinstance(bins, str):
                try:
                    candidate_edges = np.histogram_bin_edges(histogram_values, bins=bins)
                    bins = max(1, len(candidate_edges) - 1)
                except (TypeError, ValueError):
                    bins = int(np.sqrt(histogram_values.size))

            bins = max(8, int(bins))
            bins = min(bins, self.mode_histogram_max_bins, histogram_values.size)

            hist_density, bin_edges = np.histogram(
                histogram_values,
                bins=bins,
                range=(clipped_low, clipped_high),
                density=False,
            )

            if hist_density.size == 0:
                return sorted_indices[:num_samples], None

            mode_bin_idx = int(np.argmax(hist_density))
            mode_left = float(bin_edges[mode_bin_idx])
            mode_right = float(bin_edges[mode_bin_idx + 1])
            mode_center = (mode_left + mode_right) / 2

        in_band_positions = np.where(
            (sorted_distances >= clipped_low) & (sorted_distances <= clipped_high)
        )[0]
        if in_band_positions.size == 0:
            in_band_positions = np.arange(total_samples, dtype=int)

        order = np.argsort(np.abs(sorted_distances[in_band_positions] - mode_center))
        ordered_band_positions = in_band_positions[order]

        candidate_pool_size = max(
            num_samples, int(np.ceil(num_samples * self.mode_candidate_pool_multiplier))
        )
        candidate_pool_size = min(candidate_pool_size, ordered_band_positions.size)
        candidate_positions = ordered_band_positions[:candidate_pool_size]

        if candidate_positions.size == 0:
            candidate_positions = np.array(
                [int(np.argmin(np.abs(sorted_distances - mode_center)))], dtype=int
            )

        if (
            self.mode_diversity_sampling
            and features is not None
            and candidate_positions.size > num_samples
        ):
            selected_positions = self._diverse_subsample_positions(
                sorted_indices=sorted_indices,
                sorted_positions=candidate_positions,
                features=features,
                num_samples=num_samples,
            )
        else:
            selected_positions = candidate_positions[:num_samples]

        if selected_positions.size < num_samples:
            all_positions = np.arange(total_samples, dtype=int)
            mask = np.ones(total_samples, dtype=bool)
            mask[selected_positions] = False
            fallback_positions = all_positions[mask]
            if fallback_positions.size > 0:
                fallback_order = np.argsort(
                    np.abs(sorted_distances[fallback_positions] - mode_center)
                )
                needed = min(num_samples - selected_positions.size, fallback_positions.size)
                selected_positions = np.concatenate(
                    [selected_positions, fallback_positions[fallback_order[:needed]]]
                )

        selected_positions = np.array(selected_positions, dtype=int)
        selected_positions = np.unique(selected_positions)
        final_order = np.argsort(np.abs(sorted_distances[selected_positions] - mode_center))
        selected_positions = selected_positions[final_order[:num_samples]]

        mode_details = {
            "mode_center": mode_center,
            "mode_bin_start": mode_left,
            "mode_bin_end": mode_right,
            "quantile_range": [float(clipped_low), float(clipped_high)],
            "histogram_bins": int(len(bin_edges) - 1),
            "candidate_pool_size": int(candidate_pool_size),
            "diversity_sampling": bool(self.mode_diversity_sampling),
            "hist_density": hist_density.tolist(),
            "hist_bin_edges": bin_edges.tolist(),
        }

        return sorted_indices[selected_positions], mode_details

    def _sparse_band(self, distances):
        low_q, high_q = self.mode_histogram_quantile_range
        low, high = np.quantile(distances, [low_q, high_q])
        return np.flatnonzero((distances >= low) & (distances <= high)), low, high

    def _fps_dataset_indices(self, candidate_indices, features, num_samples):
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        if candidate_indices.size <= num_samples:
            return candidate_indices
        # _diverse_subsample_positions maps sorted positions through an index
        # array.  An identity ordering makes it a reusable all-candidate FPS.
        identity = np.arange(len(features), dtype=int)
        selected_positions = self._diverse_subsample_positions(
            identity, candidate_indices, features, num_samples
        )
        return identity[selected_positions]

    def _select_sparse_band_indices(
        self, distances, num_samples, features, use_fps
    ):
        candidates, low, high = self._sparse_band(distances)
        if use_fps:
            selected = self._fps_dataset_indices(candidates, features, num_samples)
        else:
            selected = np.random.permutation(candidates)[:num_samples]
        return selected, {
            "quantile_range": [float(low), float(high)],
            "candidate_pool_size": int(len(candidates)),
            "diversity_sampling": bool(use_fps),
        }

    def _select_densest_window_indices(self, distances, num_samples, features):
        """Bin-free highest-mass narrow window within the trimmed sparse band."""
        candidates, low, high = self._sparse_band(distances)
        if candidates.size == 0:
            return np.array([], dtype=int), None
        ordered = candidates[np.argsort(distances[candidates])]
        pool_size = min(
            len(ordered),
            max(
                num_samples,
                int(np.ceil(num_samples * self.mode_candidate_pool_multiplier)),
            ),
        )
        if len(ordered) <= pool_size:
            window = ordered
            start = 0
        else:
            widths = (
                distances[ordered[pool_size - 1 :]]
                - distances[ordered[: len(ordered) - pool_size + 1]]
            )
            start = int(np.argmin(widths))
            window = ordered[start : start + pool_size]
        if self.mode_diversity_sampling:
            selected = self._fps_dataset_indices(window, features, num_samples)
        else:
            selected = window[:num_samples]
        return selected, {
            "quantile_range": [float(low), float(high)],
            "candidate_pool_size": int(len(window)),
            "window_start_rank": int(start),
            "window_score_width": float(
                distances[window].max() - distances[window].min()
            ),
            "diversity_sampling": bool(self.mode_diversity_sampling),
            "bin_free": True,
        }

    def _select_inverse_cluster_indices(self, features, num_samples):
        """Closest-prior control: allocate samples inversely to cluster occupancy."""
        normalized = np.asarray(features, dtype=np.float32)
        normalized /= np.clip(
            np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12, None
        )
        num_clusters = max(2, min(int(np.sqrt(len(normalized))), len(normalized)))
        kmeans = faiss.Kmeans(
            normalized.shape[1],
            num_clusters,
            niter=25,
            nredo=1,
            seed=0,
            spherical=True,
            verbose=False,
        )
        kmeans.train(normalized)
        _, assignments = kmeans.index.search(normalized, 1)
        assignments = assignments[:, 0]
        counts = np.bincount(assignments, minlength=num_clusters)
        weights = 1.0 / np.clip(counts[assignments], 1, None)
        weights = weights / weights.sum()
        rng = np.random.default_rng(0)
        selected = rng.choice(
            len(normalized),
            size=min(num_samples, len(normalized)),
            replace=False,
            p=weights,
        )
        return selected, {
            "num_clusters": int(num_clusters),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "cluster_frequency_baseline": True,
        }

    def _diverse_subsample_positions(
        self, sorted_indices, sorted_positions, features, num_samples
    ):
        """Greedy farthest-point sampling inside a mode-centered candidate pool."""
        candidate_dataset_indices = sorted_indices[sorted_positions]
        candidate_features = np.asarray(features[candidate_dataset_indices], dtype=np.float32)

        if self.mode_diversity_normalize_features:
            norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            candidate_features = candidate_features / norms

        target_count = min(num_samples, candidate_features.shape[0])
        if target_count <= 0:
            return np.array([], dtype=int)
        if target_count == 1:
            return np.array([sorted_positions[0]], dtype=int)

        selected_local = [0]
        selected_mask = np.zeros(candidate_features.shape[0], dtype=bool)
        selected_mask[0] = True

        diff = candidate_features - candidate_features[0]
        min_sq_dist = np.einsum("ij,ij->i", diff, diff)

        for _ in range(1, target_count):
            min_sq_dist[selected_mask] = -np.inf
            next_idx = int(np.argmax(min_sq_dist))
            if selected_mask[next_idx]:
                break
            selected_local.append(next_idx)
            selected_mask[next_idx] = True

            diff = candidate_features - candidate_features[next_idx]
            sq_dist = np.einsum("ij,ij->i", diff, diff)
            min_sq_dist = np.minimum(min_sq_dist, sq_dist)

        return np.asarray(sorted_positions[np.asarray(selected_local, dtype=int)], dtype=int)
