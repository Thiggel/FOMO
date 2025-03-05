import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image
import faiss
import numpy as np
from experiment.utils.get_num_workers import get_num_workers
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


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

        # New parameters for balanced OOD selection
        self.use_clustering = args.use_clustering
        self.n_clusters = args.n_clusters
        self.class_balanced = args.class_balanced
        self.pca_dim = args.pca_dim

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
        labels = []

        with torch.no_grad():
            for batch_idx, (batch, batch_labels) in enumerate(
                tqdm(loader, desc="Extracting features")
            ):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + len(batch)
                indices.extend(range(start_idx, end_idx))
                labels.extend(batch_labels.tolist())

                # Extract features
                batch = batch.to(device=self.device, dtype=self.dtype)
                batch_features = self.feature_extractor(batch)

                if batch_features.dim() == 1:
                    batch_features = batch_features.unsqueeze(0)

                # Do NOT normalize features - we want to preserve actual distances
                features.append(batch_features.cpu())

        # Concatenate all features
        features = torch.cat(features, dim=0).numpy().astype(np.float32)
        labels = np.array(labels)

        print(f"\nFeature statistics:")
        print(f"Mean: {features.mean():.4f}")
        print(f"Std: {features.std():.4f}")
        print(f"Min: {features.min():.4f}")
        print(f"Max: {features.max():.4f}")

        return features, indices, labels

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

    def reduce_dimensions(self, features):
        """Reduce dimensionality using PCA for visualization/clustering"""
        if features.shape[1] <= self.pca_dim:
            return features

        # Use FAISS for fast PCA
        pca = faiss.PCAMatrix(features.shape[1], self.pca_dim)
        pca.train(features)
        reduced_features = np.zeros((features.shape[0], self.pca_dim), dtype=np.float32)
        pca.apply_py(features, reduced_features)

        return reduced_features

    def select_ood_samples_by_cluster(self, features, distances, labels, indices):
        """
        Select OOD samples by first clustering the feature space and then
        choosing the most OOD samples from each cluster
        """
        # Apply dimensionality reduction if needed
        if self.pca_dim > 0 and features.shape[1] > self.pca_dim:
            clustering_features = self.reduce_dimensions(features)
        else:
            clustering_features = features

        # Perform K-means clustering
        print(f"\nPerforming K-means clustering with {self.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_features)

        # Create visualization directory
        vis_dir = f"./ood_logs/{self.cycle_idx}/clustering"
        os.makedirs(vis_dir, exist_ok=True)

        # Count samples per cluster
        cluster_counts = Counter(cluster_labels)
        print("\nSamples per cluster:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"Cluster {cluster_id}: {count} samples")

        # Analyze class distribution within clusters
        class_cluster_distribution = defaultdict(lambda: defaultdict(int))
        for cluster_id, class_id in zip(cluster_labels, labels):
            class_cluster_distribution[cluster_id][class_id] += 1

        # Print top classes per cluster
        print("\nTop classes per cluster:")
        for cluster_id in sorted(class_cluster_distribution.keys()):
            class_dist = class_cluster_distribution[cluster_id]
            top_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            print(f"Cluster {cluster_id}:")
            for class_id, count in top_classes:
                class_name = (
                    self.dataset.dataset.get_class_name(class_id)
                    if hasattr(self.dataset.dataset, "get_class_name")
                    else f"Class {class_id}"
                )
                print(
                    f"  {class_name}: {count} samples ({count/cluster_counts[cluster_id]*100:.1f}%)"
                )

        # Calculate samples to select per cluster
        samples_per_cluster = self.num_ood_samples // self.n_clusters
        remainder = self.num_ood_samples % self.n_clusters

        # Adjust samples per cluster based on cluster size
        total_samples = len(features)
        adjusted_samples_per_cluster = {}
        for cluster_id, count in cluster_counts.items():
            cluster_ratio = count / total_samples
            adjusted_samples_per_cluster[cluster_id] = max(
                1, int(self.num_ood_samples * cluster_ratio)
            )

        # Ensure we're selecting exactly num_ood_samples in total
        total_adjusted = sum(adjusted_samples_per_cluster.values())
        if total_adjusted > self.num_ood_samples:
            # Remove extras from largest clusters
            sorted_clusters = sorted(
                adjusted_samples_per_cluster.items(), key=lambda x: x[1], reverse=True
            )
            excess = total_adjusted - self.num_ood_samples
            for i in range(excess):
                cluster_id = sorted_clusters[i % len(sorted_clusters)][0]
                adjusted_samples_per_cluster[cluster_id] -= 1
        elif total_adjusted < self.num_ood_samples:
            # Add to smallest clusters
            sorted_clusters = sorted(
                adjusted_samples_per_cluster.items(), key=lambda x: x[1]
            )
            deficit = self.num_ood_samples - total_adjusted
            for i in range(deficit):
                cluster_id = sorted_clusters[i % len(sorted_clusters)][0]
                adjusted_samples_per_cluster[cluster_id] += 1

        # Select the most OOD samples from each cluster
        selected_indices = []
        for cluster_id, num_to_select in adjusted_samples_per_cluster.items():
            cluster_mask = cluster_labels == cluster_id
            cluster_distances = distances[cluster_mask]
            cluster_idx = np.where(cluster_mask)[0]

            if self.class_balanced and len(np.unique(labels[cluster_mask])) > 1:
                # Apply class balancing within cluster
                class_indices = defaultdict(list)
                for i, idx in enumerate(cluster_idx):
                    class_id = labels[idx]
                    class_indices[class_id].append((idx, cluster_distances[i]))

                # Calculate samples per class
                classes_in_cluster = len(class_indices)
                samples_per_class = max(1, num_to_select // classes_in_cluster)

                # Select top samples per class
                class_selected = []
                for class_id, class_data in class_indices.items():
                    # Sort by distance (highest first)
                    sorted_class_data = sorted(
                        class_data, key=lambda x: x[1], reverse=True
                    )
                    top_n = sorted_class_data[:samples_per_class]
                    class_selected.extend([idx for idx, _ in top_n])

                # If we need more samples, get them from any class
                if len(class_selected) < num_to_select:
                    # Find remaining indices not already selected
                    remaining = [
                        idx for idx in cluster_idx if idx not in class_selected
                    ]
                    remaining_distances = [distances[idx] for idx in remaining]

                    # Sort by distance
                    sorted_remaining = [
                        x
                        for _, x in sorted(
                            zip(remaining_distances, remaining), reverse=True
                        )
                    ]
                    additional_needed = num_to_select - len(class_selected)
                    class_selected.extend(sorted_remaining[:additional_needed])

                # If we have too many, take top N overall
                if len(class_selected) > num_to_select:
                    selected_distances = [distances[idx] for idx in class_selected]
                    class_selected = [
                        x
                        for _, x in sorted(
                            zip(selected_distances, class_selected), reverse=True
                        )[:num_to_select]
                    ]

                selected_indices.extend(class_selected)
            else:
                # Just select top N by distance
                if len(cluster_distances) > 0:
                    top_n_idx = np.argsort(cluster_distances)[-num_to_select:][::-1]
                    selected_indices.extend(cluster_idx[top_n_idx])

        # Ensure we don't have duplicate indices
        selected_indices = list(set(selected_indices))

        # If we somehow don't have enough, add more from the global top OOD
        if len(selected_indices) < self.num_ood_samples:
            global_top = np.argsort(distances)[::-1]
            for idx in global_top:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= self.num_ood_samples:
                        break

        # If we have too many, take the top N by distance
        if len(selected_indices) > self.num_ood_samples:
            selected_distances = [distances[idx] for idx in selected_indices]
            selected_indices = [
                x
                for _, x in sorted(
                    zip(selected_distances, selected_indices), reverse=True
                )[: self.num_ood_samples]
            ]

        # Map back to original dataset indices
        ood_indices = [indices[i] for i in selected_indices]

        # Print class distribution of selected samples
        selected_classes = [labels[i] for i in selected_indices]
        class_counts = Counter(selected_classes)
        print("\nClass distribution of selected OOD samples:")
        for class_id, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            class_name = (
                self.dataset.dataset.get_class_name(class_id)
                if hasattr(self.dataset.dataset, "get_class_name")
                else f"Class {class_id}"
            )
            print(
                f"{class_name}: {count} samples ({count/len(selected_indices)*100:.1f}%)"
            )

        # Visualize clustering results if possible
        if clustering_features.shape[1] >= 2:
            plt.figure(figsize=(12, 10))
            plt.scatter(
                clustering_features[:, 0],
                clustering_features[:, 1],
                c=cluster_labels,
                cmap="viridis",
                alpha=0.5,
                s=20,
            )
            plt.scatter(
                clustering_features[selected_indices, 0],
                clustering_features[selected_indices, 1],
                c="red",
                s=50,
                marker="x",
                label="Selected OOD samples",
            )
            plt.title(
                f"Feature Space Clustering and OOD Sample Selection (Cycle {self.cycle_idx})"
            )
            plt.xlabel("PCA Dimension 1")
            plt.ylabel("PCA Dimension 2")
            plt.legend()
            plt.savefig(f"{vis_dir}/clustering_viz.png", dpi=300)
            plt.close()

            # Create visualization showing class distribution
            unique_classes = np.unique(labels)
            plt.figure(figsize=(12, 10))
            for class_id in unique_classes:
                mask = labels == class_id
                if mask.sum() > 0:  # Only plot if we have samples of this class
                    class_name = (
                        self.dataset.dataset.get_class_name(class_id)
                        if hasattr(self.dataset.dataset, "get_class_name")
                        else f"Class {class_id}"
                    )
                    plt.scatter(
                        clustering_features[mask, 0],
                        clustering_features[mask, 1],
                        alpha=0.5,
                        s=20,
                        label=(
                            class_name if mask.sum() > 20 else None
                        ),  # Only show legend for classes with enough samples
                    )

            # Highlight selected samples
            plt.scatter(
                clustering_features[selected_indices, 0],
                clustering_features[selected_indices, 1],
                c="red",
                s=50,
                marker="x",
                label="Selected OOD samples",
            )
            plt.title(f"Class Distribution in Feature Space (Cycle {self.cycle_idx})")
            plt.xlabel("PCA Dimension 1")
            plt.ylabel("PCA Dimension 2")
            plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=1)
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/class_distribution.png", dpi=300)
            plt.close()

        return ood_indices

    def ood(self):
        """Identify the most out-of-distribution samples"""
        # Extract features (without normalization)
        features, indices, labels = self.extract_features()

        # Compute distances
        distances = self.compute_knn_distances(features)

        if self.use_clustering:
            return self.select_ood_samples_by_cluster(
                features, distances, labels, indices
            )

        # Traditional global OOD selection
        # Calculate number of samples to select
        num_samples = (
            int(self.num_ood_samples * len(self.dataset))
            if isinstance(self.num_ood_samples, float)
            else self.num_ood_samples
        )

        # Get indices of points with largest mean k-NN distances
        top_indices = np.argsort(distances)[-num_samples:][::-1]

        # Print the top distances for analysis
        print("\nTop OOD samples:")
        for idx in top_indices[:10]:  # Just print the first 10 for brevity
            print(f"Index {indices[idx]}: mean kNN distance = {distances[idx]:.4f}")
            if isinstance(self.dataset[indices[idx]][0], torch.Tensor):
                print(f"Point coordinates: {self.dataset[indices[idx]][0]}")

        # Print class distribution of selected samples
        selected_classes = [labels[i] for i in top_indices]
        class_counts = Counter(selected_classes)
        print("\nClass distribution of selected OOD samples:")
        for class_id, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            class_name = (
                self.dataset.dataset.get_class_name(class_id)
                if hasattr(self.dataset.dataset, "get_class_name")
                else f"Class {class_id}"
            )
            print(f"{class_name}: {count} samples ({count/len(top_indices)*100:.1f}%)")

        # Map to original dataset indices
        ood_indices = [indices[i] for i in top_indices]

        # Save results and visualizations
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

        np.save(f"./ood_logs/{self.cycle_idx}/distances.npy", distances)

        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, _ = self.dataset[ood_indices[i]]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                distance = distances[top_indices[i]]
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}.jpg"
                save_image(image, image_path)

        return ood_indices
