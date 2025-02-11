import sys
import io
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Subset, random_split, Dataset
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from experiment.ood.ood import OOD
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms
import copy
import matplotlib.pyplot as plt

import os
import pickle

from experiment.dataset.ImageStorage import ImageStorage


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

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            if not self.args.test_mode and os.path.exists(
                self.checkpoint_callback.best_model_path
            ):
                output_path = (
                    self.checkpoint_callback.best_model_path
                    + "_fp32.pt".replace(":", "_").replace(" ", "_")
                )

                convert_zero_checkpoint_to_fp32_state_dict(
                    self.checkpoint_callback.best_model_path,
                    output_path,
                )

                self.ssl_method.load_state_dict(torch.load(output_path)["state_dict"])

        return self.finetune() if self.args.finetune else {}

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

    def get_class_indices_map(self, dataset):
        """Efficiently create a mapping of class labels to their indices"""
        class_to_indices = {}
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1024,
            num_workers=self.datamodule.num_workers,
            pin_memory=True,
        )

        available_indices = list(self.original_indices - self.added_indices)
        for batch_idx, (_, labels) in enumerate(
            tqdm(dataloader, desc="Creating class index map")
        ):
            for i, label in enumerate(labels):
                idx = batch_idx * dataloader.batch_size + i
                if idx in available_indices:
                    label_int = label.item()
                    if label_int not in class_to_indices:
                        class_to_indices[label_int] = []
                    class_to_indices[label_int].append(idx)

        return class_to_indices

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
        try:
            initial_size = len(self.datamodule.train_dataset)
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
                    },
                )
                cycle_trainer_args["strategy"] = strategy
                cycle_trainer_args["accelerator"] = "cuda"
                cycle_trainer_args["devices"] = "auto"

            trainer = L.Trainer(**cycle_trainer_args)
            trainer.fit(model=self.ssl_method, datamodule=self.datamodule)

            if not self.args.ood_augmentation:
                return

            ssl_transform = copy.deepcopy(
                self.datamodule.train_dataset.dataset.transform
            )
            self.datamodule.train_dataset.dataset.transform = self.transform

            train_dataset = Subset(
                self.datamodule.train_dataset, list(range(self.initial_train_ds_size))
            )

            num_ood_test = int(self.ood_test_split * len(train_dataset))
            num_ood_train = len(train_dataset) - num_ood_test
            ood_train_dataset, ood_test_dataset = random_split(
                train_dataset, [num_ood_train, num_ood_test]
            )

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

                # Get efficient mapping of classes to indices
                class_to_indices = self.get_class_indices_map(train_dataset)
                new_indices = []

                # Sample from each OOD class efficiently
                for cls, num_samples in tqdm(
                    num_samples_per_class.items(), desc="Sampling from OOD classes"
                ):
                    if cls in class_to_indices:
                        available_class_indices = class_to_indices[cls]
                        sampled_count = min(num_samples, len(available_class_indices))
                        if sampled_count > 0:
                            sampled_indices = np.random.choice(
                                available_class_indices,
                                size=sampled_count,
                                replace=False,
                            ).tolist()
                            new_indices.extend(sampled_indices)

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
            else:
                indices_to_be_augmented = (
                    self.get_ood_indices(ood_train_dataset, ood_test_dataset, cycle_idx)
                    if self.args.use_ood
                    else self.get_random_indices(ood_train_dataset)
                )

                # Get labels for the OOD samples
                ood_samples = Subset(ood_train_dataset, indices_to_be_augmented)
                ood_labels = [
                    ood_samples[i][1] for i in range(len(ood_samples))
                ]  # Get labels

                # Original diffusion-based augmentation code
                diffusion_pipe = self.initialize_model(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.generate_new_data(
                    ood_samples,
                    pipe=diffusion_pipe,
                    batch_size=self.args.sd_batch_size,
                    save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
                )
                # Add the generated image count and labels to the dataset
                self.datamodule.train_dataset.dataset.add_generated_images(
                    cycle_idx, len(ood_samples), ood_labels  # Pass the labels here
                )

                self.datamodule.train_dataset = torch.utils.data.Subset(
                    self.datamodule.train_dataset.dataset,
                    list(
                        range(len(self.datamodule.train_dataset.dataset))
                    ),  # Get all indices including new data
                )

            # Reset transforms
            self.datamodule.train_dataset.dataset.transform = ssl_transform

            if self.args.ood_augmentation:
                final_size = len(self.datamodule.train_dataset)
                assert final_size > initial_size, (
                    f"Dataset size did not increase after augmentation. "
                    f"Initial size: {initial_size}, Final size: {final_size}"
                )

                if cycle_idx > 0:
                    previous_cycle_size = getattr(
                        self, "previous_cycle_size", initial_size
                    )
                    assert final_size > previous_cycle_size, (
                        f"Cycle {cycle_idx} size ({final_size}) not larger than "
                        f"cycle {cycle_idx-1} size ({previous_cycle_size})"
                    )
                self.previous_cycle_size = final_size

        finally:
            # Clean up resources
            self._cleanup_cycle_resources(trainer)

    def _cleanup_cycle_resources(self, trainer):
        """Clean up resources after each cycle."""
        # Clear DataLoaders
        self.datamodule.set_dataloaders_none()

        # Clean up trainer
        if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "cleanup"):
            trainer.strategy.cleanup()

        # Force release CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_num_samples_to_generate(self) -> int:
        return int(
            self.args.pct_ood * self.ood_test_split * len(self.datamodule.train_dataset)
        )

    def get_random_indices(self, ood_train_dataset) -> list:
        num_samples = self.get_num_samples_to_generate()
        return torch.randperm(len(ood_train_dataset))[:num_samples].tolist()

    def get_ood_indices(self, ood_train_dataset, ood_test_dataset, cycle_idx) -> list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_method.to(device)
        self.ssl_method.model.to(dtype=self.ssl_method.dtype)
        ood = OOD(
            args=self.args,
            train=ood_train_dataset,
            test=ood_test_dataset,
            feature_extractor=self.ssl_method.model.extract_features,
            cycle_idx=cycle_idx,
            device=self.ssl_method.device,
            dtype=self.ssl_method.dtype,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        return ood_indices

    def save_class_distribution(self, cycle_idx: int) -> None:
        """
        Save the class distribution for current cycle, accounting for both original and augmented data.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_counts = torch.zeros(
            self.datamodule.train_dataset.dataset.num_classes, device=device
        )

        loader = torch.utils.data.DataLoader(
            self.datamodule.train_dataset,
            batch_size=2048,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        with torch.no_grad():
            for _, labels in tqdm(
                loader, desc=f"Counting samples for cycle {cycle_idx}"
            ):
                labels = labels.to(device)
                counts = torch.bincount(
                    labels, minlength=self.datamodule.train_dataset.dataset.num_classes
                )
                class_counts += counts

            # Print some validation info
            print(f"\nCycle {cycle_idx} distribution:")
            print(f"Total samples: {class_counts.sum().item()}")
            print(f"Per-class counts: {class_counts.cpu().tolist()}")
            print(f"Non-zero classes: {(class_counts > 0).sum().item()}\n")

        # Save distribution
        save_path = f"{os.environ['BASE_CACHE_DIR']}/class_distributions/{self.checkpoint_filename}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(class_counts.cpu(), f"{save_path}/dist_cycle_{cycle_idx}.pt")

    def pretrain_imbalanced(self) -> None:
        """
        1. Fit for n_epochs_per_cycle epochs,
        2. Use validation set to determine OOD samples
        3. Generate augmentations for OOD samples
        4. Restart
        """
        visualization_dir = (
            f"visualizations/class_distributions/{self.checkpoint_filename}"
        )
        os.makedirs(visualization_dir, exist_ok=True)

        for cycle_idx in range(self.max_cycles):
            print(f"Run {self.run_idx + 1}/{self.args.num_runs}")
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")
            self.pretrain_cycle(cycle_idx)

            self.save_class_dist(cycle_idx)

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.get_benchmarks(
            self.args.finetuning_benchmarks
        )
        results = {}

        self.trainer_args.pop("callbacks")

        torch.multiprocessing.set_sharing_strategy("file_system")

        for benchmark in benchmarks:
            print("\n -- Finetuning benchmark:", benchmark.__name__, "--\n")

            transform = transforms.Compose(
                [
                    transforms.Resize((self.args.crop_size, self.args.crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # dataloader is already handled fine here because each loop should set the past loader to None.
            finetuner = benchmark(
                model=self.ssl_method.model,
                lr=self.args.lr,
                transform=transform,
                crop_size=self.args.crop_size,
            )

            self.trainer_args["max_epochs"] = finetuner.max_epochs

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                verbose=True,
                min_delta=0.0001,
            )

            if not "KNN" in benchmark.__name__:
                self.trainer_args["callbacks"] = [early_stop_callback]
            else:
                self.trainer_args["callbacks"] = []

            self.trainer_args["max_time"] = {
                "minutes": 25,
            }

            self.trainer_args["accumulate_grad_batches"] = 1

            strategy = DeepSpeedStrategy(
                config={
                    "train_batch_size": 64 * torch.cuda.device_count(),
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": 2,
                        "offload_optimizer": {"device": "cpu", "pin_memory": True},
                        "offload_param": {"device": "cpu", "pin_memory": True},
                    },
                },
            )
            self.trainer_args["strategy"] = strategy
            self.trainer_args.pop("accelerator", None)

            trainer = L.Trainer(**self.trainer_args)

            trainer.fit(model=finetuner)

            finetuning_results = trainer.test(model=finetuner)[0]

            results = {**results, **finetuning_results}

        return results

    def initialize_model(self, device):
        """
        Load the model first to ensure better flow
        """
        if device == "cpu":
            pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip"
            )
        else:
            pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip",
                torch_dtype=torch.float16,
                variation="bf16",
            )
        pipe = pipe.to(device)
        return pipe

    def generate_new_data(
        self, ood_samples, pipe, save_subfolder, batch_size=4, nr_to_gen=1
    ) -> None:
        """
        Generate new data using the diffusion model.
        """
        cycle_idx = int(save_subfolder.split("/")[-1])
        image_storage = ImageStorage(self.args.additional_data_path)

        # Create a simple transform for denormalization
        denorm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )

        k = 0
        for b_start in tqdm(
            range(0, len(ood_samples), batch_size), desc="Generating New Data..."
        ):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Get and denormalize batch
            batch = [
                ood_samples[i + b_start][0]
                for i in range(min(len(ood_samples) - b_start, batch_size))
            ]
            batch = [denorm(img) for img in batch]

            # Generate images
            v_imgs = pipe(batch, num_images_per_prompt=nr_to_gen).images

            # Save batch
            image_storage.save_batch(v_imgs, cycle_idx, k)
            k += len(v_imgs)

            sys.stdout = old_stdout
