import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image
import faiss
import numpy as np
from experiment.utils.get_num_workers import get_num_workers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wandb
import matplotlib

matplotlib.use("Agg")  # Use Agg backend to avoid display issues


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
        self.args = args

        # Create directories for visualizations
        self.viz_dir = f"{os.environ.get('BASE_CACHE_DIR', '.')}/visualizations/ood_viz/cycle_{cycle_idx}"
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(f"./ood_logs/{self.cycle_idx}/images", exist_ok=True)

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
        features = torch.cat(features, dim=0)
        labels = torch.tensor(labels)

        print(f"\nFeature statistics:")
        print(f"Mean: {features.mean():.4f}")
        print(f"Std: {features.std():.4f}")
        print(f"Min: {features.min():.4f}")
        print(f"Max: {features.max():.4f}")

        torch.cuda.empty_cache()

        return features.numpy().astype(np.float32), np.array(indices), labels.numpy()

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

    def visualize_latent_space(self, features, knn_distances, labels, indices):
        """
        Create 2D PCA visualization of the latent space showing clusters
        and which points are most OOD
        """
        print("\nCreating PCA visualization of latent space...")

        # Use PCA for visualization only (not for clustering)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Create the visualization figure
        plt.figure(figsize=(12, 10))

        # Normalize distances for better visualization
        normalized_distances = (knn_distances - knn_distances.min()) / (
            knn_distances.max() - knn_distances.min()
        )

        # Create scatter plot with color intensity representing OOD score
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=normalized_distances,
            cmap="viridis",
            alpha=0.7,
            s=30,
        )

        # Highlight top OOD points
        num_ood = min(100, len(features))
        top_ood_indices = np.argsort(knn_distances)[-num_ood:][::-1]
        plt.scatter(
            features_2d[top_ood_indices, 0],
            features_2d[top_ood_indices, 1],
            facecolors="none",
            edgecolors="red",
            s=80,
            linewidth=1.5,
            label=f"Top {num_ood} OOD points",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("OOD Score (Normalized K-NN Distance)")

        # Add labels and title
        plt.title(f"Latent Space Visualization (Cycle {self.cycle_idx})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(loc="upper right")
        plt.tight_layout()

        # Save the figure as PDF
        pdf_path = f"{self.viz_dir}/latent_space_cycle_{self.cycle_idx}.pdf"
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

        # Save the visualization data for future reference
        viz_data = {
            "pca_coords": features_2d,
            "knn_distances": knn_distances,
            "labels": labels,
            "indices": indices,
            "top_ood_indices": top_ood_indices,
        }
        torch.save(
            viz_data, f"{self.viz_dir}/latent_space_data_cycle_{self.cycle_idx}.pt"
        )

        # If using wandb, log the visualization
        if self.args.logger and not self.args.test_mode:
            try:
                wandb.log(
                    {
                        f"latent_space/cycle_{self.cycle_idx}": wandb.Image(pdf_path),
                        "cycle": self.cycle_idx,
                    }
                )

                # Create a second visualization showing class distribution in latent space
                self.visualize_class_distribution_in_latent_space(
                    features_2d, labels, indices
                )
            except Exception as e:
                print(f"Warning: Failed to log visualization to wandb: {e}")

        plt.close()
        return pdf_path, viz_data

    def visualize_class_distribution_in_latent_space(
        self, features_2d, labels, indices
    ):
        """Create a class-colored visualization of the latent space"""
        # Get the number of unique classes
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)

        # Create a new figure
        plt.figure(figsize=(14, 10))

        # Use a colormap that can handle many classes
        cmap = plt.cm.get_cmap("tab20", num_classes)

        # Plot each class with a different color
        for i, class_idx in enumerate(unique_classes):
            mask = labels == class_idx
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[cmap(i)],
                label=f"Class {class_idx}",
                alpha=0.7,
                s=30,
            )

        # Add labels and title
        plt.title(f"Class Distribution in Latent Space (Cycle {self.cycle_idx})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        # Add a legend (outside of the plot if many classes)
        if num_classes > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.legend(loc="upper right")

        plt.tight_layout()

        # Save the figure
        class_viz_path = f"{self.viz_dir}/class_distribution_cycle_{self.cycle_idx}.pdf"
        plt.savefig(class_viz_path, format="pdf", dpi=300, bbox_inches="tight")

        # Log to wandb
        if self.args.logger and not self.args.test_mode:
            try:
                wandb.log(
                    {
                        f"class_distribution_latent/cycle_{self.cycle_idx}": wandb.Image(
                            class_viz_path
                        ),
                        "cycle": self.cycle_idx,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to log class visualization to wandb: {e}")

        plt.close()

    def ood(self):
        """Identify the most out-of-distribution samples"""
        # Extract features (without normalization)
        features, indices, labels = self.extract_features()

        # Compute distances using KNN in the full feature space (no PCA dimensionality reduction)
        distances = self.compute_knn_distances(features)

        # Create visualizations before selecting OOD samples
        self.visualize_latent_space(features, distances, labels, indices)

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
        for idx in top_indices:
            print(f"Index {indices[idx]}: mean kNN distance = {distances[idx]:.4f}")
            if isinstance(self.dataset[indices[idx]][0], torch.Tensor):
                print(f"Label: {labels[idx]}")

        # Map to original dataset indices
        ood_indices = [indices[i] for i in top_indices]

        # Save results
        np.save(f"./ood_logs/{self.cycle_idx}/distances.npy", distances)

        # Visualize a few of the OOD samples
        num_vis = min(10, len(ood_indices))
        for i in range(num_vis):
            image, label = self.dataset[ood_indices[i]]
            if isinstance(image, torch.Tensor) and len(image.shape) in [3, 4]:
                distance = distances[top_indices[i]]
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}_label_{label}.jpg"
                save_image(image, image_path)

        return ood_indices
