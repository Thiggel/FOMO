import os
import lightning.pytorch as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.ImbalancedImageNet import DummyImageNet, ImbalancedImageNet
from experiment.dataset.imbalancedness.ImbalanceMethods import (
    ImbalanceMethods,
    ImbalanceMethod,
)
from torchvision import transforms
from datasets import load_dataset
import random

from experiment.utils.get_num_workers import get_num_workers


class ImbalancedTraining:
    def __init__(
        self,
        args: dict,
        trainer_args: dict,
        ssl_method: L.LightningModule,
        datamodule: L.LightningDataModule,
        checkpoint_callback: L.Callback,
        checkpoint_filename: str,
        save_class_distribution: bool = False,
        run_idx: int = 0,
    ):
        self.args = args
        self.run_idx = run_idx
        self.trainer_args = trainer_args
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_filename = checkpoint_filename
        self.save_class_distribution = save_class_distribution
        self.n_epochs_per_cycle = args.n_epochs_per_cycle
        self.max_cycles = args.max_cycles
        self.ood_test_split = args.ood_test_split
        self.transform = transforms.Compose(
            [
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.datamodule is not None:
            self.initial_train_ds_size = len(self.datamodule.train_dataset)
            # Track which indices have been added back to prevent duplicates
            self.added_indices = set()
            # Store original dataset indices for sampling
            self.original_indices = set(range(self.initial_train_ds_size))

    def get_class_of_index(self, dataset, idx):
        """Helper function to get class label for a given index"""
        _, label = dataset[idx]
        return label

    def get_ood_classes(self, ood_indices, dataset):
        """Get class distribution of OOD points"""
        class_counts = {}
        for idx in ood_indices:
            label = self.get_class_of_index(dataset, idx)
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def sample_from_class(self, dataset, target_class, exclude_indices, num_samples):
        """Sample indices from a specific class, excluding already added indices"""
        available_indices = []
        for idx in self.original_indices - self.added_indices:
            if self.get_class_of_index(dataset, idx) == target_class:
                available_indices.append(idx)

        # If we don't have enough samples from the target class,
        # return all available ones
        if len(available_indices) <= num_samples:
            return available_indices

        return np.random.choice(
            available_indices, size=num_samples, replace=False
        ).tolist()

    def sample_uniformly(self, num_samples):
        """Sample uniformly from remaining indices when class-based sampling is not possible"""
        available_indices = list(self.original_indices - self.added_indices)
        if (
            not available_indices
        ):  # If no indices left, sample from all original indices
            available_indices = list(self.original_indices)

        return np.random.choice(
            available_indices,
            size=min(num_samples, len(available_indices)),
            replace=False,
        ).tolist()

    def pretrain_cycle(self, cycle_idx) -> None:
        """
        1. Fit for n epochs
        2. assess OOD samples
        3. generate new data for OOD samples or add back removed data
        """
        cycle_trainer_args = self.trainer_args.copy()

        # Ensure strategy is properly configured for each cycle
        if torch.cuda.is_available():
            strategy = DeepSpeedStrategy(
                config={
                    "train_batch_size": self.args.batch_size
                    * self.args.grad_acc_steps
                    * torch.cuda.device_count(),
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": 2,
                        "offload_optimizer": {"device": "cpu", "pin_memory": True},
                        "offload_param": {"device": "cpu", "pin_memory": True},
                    },
                }
            )
            cycle_trainer_args["strategy"] = strategy
            cycle_trainer_args["accelerator"] = "cuda"
            cycle_trainer_args["devices"] = "auto"

        trainer = L.Trainer(**cycle_trainer_args)
        trainer.fit(model=self.ssl_method, datamodule=self.datamodule)
        self.datamodule.set_dataloaders_none()

        if not self.args.ood_augmentation:
            return

        ssl_transform = copy.deepcopy(self.datamodule.train_dataset.dataset.transform)
        self.datamodule.train_dataset.dataset.transform = self.transform

        train_dataset = Subset(
            self.datamodule.train_dataset, list(range(self.initial_train_ds_size))
        )

        num_ood_test = int(self.ood_test_split * len(train_dataset))
        num_ood_train = len(train_dataset) - num_ood_test
        ood_train_dataset, ood_test_dataset = random_split(
            train_dataset, [num_ood_train, num_ood_test]
        )

        # Get OOD indices using the existing method
        indices_to_be_augmented = (
            self.get_ood_indices(ood_train_dataset, ood_test_dataset, cycle_idx)
            if self.args.use_ood
            else self.get_random_indices(ood_train_dataset)
        )

        if self.args.remove_diffusion:
            # Get the class distribution of OOD points
            ood_class_dist = self.get_ood_classes(
                indices_to_be_augmented, ood_train_dataset
            )

            # Calculate number of samples to generate per class
            num_samples_per_class = {
                cls: int(count * self.args.pct_ood * self.ood_test_split)
                for cls, count in ood_class_dist.items()
            }

            new_indices = []
            # First try to sample from each OOD class
            for cls, num_samples in num_samples_per_class.items():
                class_indices = self.sample_from_class(
                    train_dataset, cls, self.added_indices, num_samples
                )
                new_indices.extend(class_indices)

            # If we still need more samples, sample uniformly
            total_samples_needed = int(
                self.args.pct_ood * self.ood_test_split * self.initial_train_ds_size
            )
            if len(new_indices) < total_samples_needed:
                additional_indices = self.sample_uniformly(
                    total_samples_needed - len(new_indices)
                )
                new_indices.extend(additional_indices)

            # Update the set of added indices
            self.added_indices.update(new_indices)

            # Add the selected indices to the dataset
            self.datamodule.add_samples_by_index(new_indices)

            print(f"Added {len(new_indices)} samples back to the training set")
            return

        # Original diffusion-based augmentation code
        ood_samples = Subset(ood_train_dataset, indices_to_be_augmented)
        if cycle_idx < self.max_cycles - 1:
            self.datamodule.train_dataset.dataset.transform = None
            diffusion_pipe = self.initialize_model(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.generate_new_data(
                ood_samples,
                pipe=diffusion_pipe,
                batch_size=self.args.sd_batch_size,
                save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
            )
            self.datamodule.train_dataset.dataset.transform = ssl_transform
