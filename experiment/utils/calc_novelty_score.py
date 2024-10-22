import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np


def calc_novelty_score(args, ssl_method):
    """
    Calculate novelty scores for different datasets using KNN distances.

    Args:
        args: Arguments containing data paths and parameters
        ssl_method: Trained SSL model (SimCLR) for feature extraction

    Returns:
        dict: Dictionary containing novelty scores for each dataset
    """
    # Set model to evaluation mode
    ssl_method.eval()

    # Define common transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Define datasets
    datasets_dict = {
        "birds": datasets.ImageFolder(root=args.birds_path, transform=transform),
        "cifar10": datasets.CIFAR10(
            root=args.data_path, train=False, transform=transform, download=True
        ),
        "cifar100": datasets.CIFAR100(
            root=args.data_path, train=False, transform=transform, download=True
        ),
        "flowers": datasets.ImageFolder(root=args.flowers_path, transform=transform),
    }

    # Initialize results dictionary
    novelty_scores = {}

    # Number of neighbors for KNN
    k = 5

    # Process each dataset
    with torch.no_grad():
        for dataset_name, dataset in datasets_dict.items():
            # Create dataloader
            dataloader = DataLoader(
                dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
            )

            # Extract features
            features = []
            for batch, _ in dataloader:
                batch = batch.to(ssl_method.device)
                feat = ssl_method.model(batch)
                features.append(feat.cpu().numpy())

            # Concatenate all features
            features = np.concatenate(features, axis=0)

            # Fit KNN
            knn = NearestNeighbors(
                n_neighbors=k + 1
            )  # k+1 because first neighbor is self
            knn.fit(features)

            # Calculate distances
            distances, _ = knn.kneighbors(features)

            # Calculate novelty score (average distance to k nearest neighbors)
            # Exclude the first neighbor (which is self) by using distances[:, 1:]
            novelty_score = np.mean(np.sum(distances[:, 1:], axis=1))

            # Store result
            novelty_scores[f"{dataset_name}_novelty_score"] = float(novelty_score)

    return novelty_scores


def process_args_example():
    """Example of expected args structure"""

    class Args:
        def __init__(self):
            self.data_path = "./data"
            self.birds_path = "./data/birds"
            self.flowers_path = "./data/flowers"

    return Args()


# Example usage:
# args = process_args_example()
# model = SimCLR(...)  # Your trained SSL model
# scores = calc_novelty_score(args, model)
# print(scores)
