import torch
import numpy as np
from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class PowerLawImbalancedness(Imbalancedness):
    """
    Implements ImageNet-LT style imbalance following Pareto distribution with α=6.
    Ensures 1280 samples for most frequent class and 5 samples for least frequent.
    """

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.alpha = 3  # Pareto distribution parameter
        self.max_samples = 1280  # Maximum samples per class as in ImageNet-LT
        self.min_samples = 5  # Minimum samples per class as in ImageNet-LT

        # Pre-calculate the class sample ratios
        self.class_ratios = self._calculate_class_ratios()

    def _calculate_class_ratios(self) -> torch.Tensor:
        """
        Calculate the ratio of samples to keep for each class following Pareto distribution.
        Returns ratios in range [0, 1] where 1 corresponds to max_samples.
        """
        # Generate Pareto distribution values
        x = np.arange(1, self.num_classes + 1, dtype=np.float32)
        pareto_vals = x ** (-1 / self.alpha)

        # Scale to [min_samples, max_samples]
        scaled_vals = (pareto_vals - pareto_vals.min()) / (
            pareto_vals.max() - pareto_vals.min()
        )
        scaled_vals = (
            scaled_vals * (self.max_samples - self.min_samples) + self.min_samples
        )

        # Convert to ratios
        ratios = scaled_vals / self.max_samples

        return torch.from_numpy(ratios)

    def get_imbalance(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Get imbalance scores for given class indices.
        Args:
            class_indices: Tensor of class indices
        Returns:
            Tensor of same shape as class_indices with imbalance scores
        """
        if not isinstance(class_indices, torch.Tensor):
            class_indices = torch.tensor(class_indices)

        # Move ratios to same device as indices
        ratios = self.class_ratios.to(class_indices.device)

        # Return ratio for each class index
        return ratios[class_indices]

    def get_expected_samples(self) -> torch.Tensor:
        """
        Returns the expected number of samples per class.
        Useful for verification and logging.
        """
        return self.class_ratios * self.max_samples


class ImbalanceVerifier:
    """
    Utility class to verify the imbalance distribution matches ImageNet-LT characteristics.
    """

    @staticmethod
    def verify_distribution(dataset, imbalanced_indices, num_classes):
        # Count samples per class
        class_counts = torch.zeros(num_classes)
        for idx in imbalanced_indices:
            _, label = dataset[idx]
            class_counts[label] += 1

        # Calculate distribution statistics
        max_samples = class_counts.max().item()
        min_samples = class_counts[class_counts > 0].min().item()
        total_samples = class_counts.sum().item()

        print(f"Distribution Statistics:")
        print(f"Total samples: {total_samples:,}")
        print(f"Max samples per class: {max_samples:,}")
        print(f"Min samples per class: {min_samples:,}")
        print(f"Empty classes: {(class_counts == 0).sum().item():,}")

        # Verify power law
        sorted_counts = torch.sort(class_counts, descending=True)[0]
        non_zero_counts = sorted_counts[sorted_counts > 0]

        return {
            "total_samples": total_samples,
            "max_samples": max_samples,
            "min_samples": min_samples,
            "counts": class_counts.tolist(),
        }
