"""End-to-end low-shot transfer on a deterministic CIFAR-100-LT target."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .CIFARDatasets import get_cifar_dataset
from .DatasetRoots import get_data_root
from .TransferLearningBenchmark import TransferLearningBenchmark


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=int)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])


class CIFAR100LTFineTune(TransferLearningBenchmark):
    """CIFAR-100 with the standard exponential 100:1 long-tail profile."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        *args,
        seed: int = 42,
        **kwargs,
    ):
        self.seed = int(seed)
        super().__init__(
            model=model,
            lr=lr,
            transform=transform,
            num_classes=100,
            *args,
            **kwargs,
        )
        self._log_per_class_accuracy = True
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        root = get_data_root()
        train_base = get_cifar_dataset(
            num_classes=100, root=root, train=True, transform=self.transform
        )
        test_dataset = get_cifar_dataset(
            num_classes=100, root=root, train=False, transform=self.transform
        )
        targets = _targets(train_base)
        rng = np.random.default_rng(self.seed)
        train_indices, val_indices = [], []
        class_counts = {}
        for class_id in range(100):
            candidates = np.flatnonzero(targets == class_id)
            rng.shuffle(candidates)
            # Common CIFAR-LT exponential profile: 500 -> 5 examples.
            desired = max(5, int(round(500 * (0.01 ** (class_id / 99.0)))))
            selected = candidates[: min(desired, len(candidates))]
            val_count = max(1, int(round(0.1 * len(selected))))
            val_indices.extend(selected[:val_count].tolist())
            train_indices.extend(selected[val_count:].tolist())
            class_counts[class_id] = int(len(selected) - val_count)

        self.target_train_class_counts = class_counts
        self.many_shot_classes = [
            class_id for class_id, count in class_counts.items() if count > 100
        ]
        self.medium_shot_classes = [
            class_id for class_id, count in class_counts.items() if 20 <= count <= 100
        ]
        self.few_shot_classes = [
            class_id for class_id, count in class_counts.items() if count < 20
        ]
        return (
            Subset(train_base, train_indices),
            Subset(train_base, val_indices),
            test_dataset,
        )

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)
