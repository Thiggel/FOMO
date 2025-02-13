import sys
import io
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Subset, random_split, Dataset, DataLoader
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
        run_idx: int = 0,
    ):
        self.args = args
        self.run_idx = run_idx
        self.trainer_args = trainer_args
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_filename = checkpoint_filename
        self.save_class_distribution = self.args.save_class_distribution
        self.n_epochs_per_cycle = args.n_epochs_per_cycle
        self.max_cycles = args.max_cycles
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
            self.added_indices = set()
            self.original_indices = set(range(self.initial_train_ds_size))

        self.num_workers = self.datamodule.num_workers

        if self.save_class_distribution:
            self.save_class_dist(0)

    def get_batch_labels(self, dataset, indices):
        """Get labels for multiple indices efficiently using DataLoader"""
        # If dataset is already a Subset, we need to map our indices through its indices
        actual_indices = [dataset.indices[i] for i in indices]
        # Create new subset from the base dataset
        subset = Subset(dataset.dataset, actual_indices)

        loader = DataLoader(
            subset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        labels = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for batch in loader:
            _, batch_labels = batch  # Each batch is (tensor_batch, labels_batch)
            labels.append(batch_labels.to(device))

        return torch.cat(labels)

    def get_ood_classes(self, ood_indices, dataset):
        """Get class distribution using GPU-accelerated operations"""
        labels = self.get_batch_labels(dataset, ood_indices)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        return dict(zip(unique_labels.cpu().tolist(), counts.cpu().tolist()))

    def get_class_indices_map(self, dataset):
        """Efficiently create a mapping of class labels to their indices"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        available_indices = torch.tensor(
            list(self.original_indices - self.added_indices), device=device
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        class_to_indices = {}
        for batch_idx, (_, labels) in enumerate(
            tqdm(dataloader, desc="Creating class index map")
        ):
            labels = labels.to(device)
            batch_start_idx = batch_idx * dataloader.batch_size
            batch_indices = available_indices[
                batch_start_idx : batch_start_idx + len(labels)
            ]

            # Group indices by class efficiently
            for label in labels.unique():
                label_mask = labels == label
                if label.item() not in class_to_indices:
                    class_to_indices[label.item()] = []
                class_to_indices[label.item()].extend(
                    batch_indices[label_mask].tolist()
                )

        return class_to_indices

    def pretrain_cycle(self, cycle_idx) -> None:
        try:
            # Log initial dataset size
            initial_size = len(self.datamodule.train_dataset)
            print(f"\nCycle {cycle_idx} - Initial dataset size: {initial_size}")

            cycle_trainer_args = self.trainer_args.copy()

            if torch.cuda.is_available():
                strategy = DeepSpeedStrategy(
                    config={
                        "train_batch_size": self.args.train_batch_size
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
                print("OOD augmentation disabled, skipping generation")
                return

            ssl_transform = copy.deepcopy(
                self.datamodule.train_dataset.dataset.transform
            )
            self.datamodule.train_dataset.dataset.transform = self.transform

            train_dataset = Subset(
                self.datamodule.train_dataset, list(range(self.initial_train_ds_size))
            )

            # Log OOD selection method
            print(
                f"\nSelecting OOD samples using {'OOD detection' if self.args.use_ood else 'random selection'}"
            )

            indices_to_be_augmented = (
                self.get_ood_indices(train_dataset, cycle_idx)
                * self.args.num_generations_per_ood_sample
                if self.args.use_ood
                else self.get_random_indices(train_dataset)
            )

            print(f"Selected {len(indices_to_be_augmented)} samples for augmentation")

            if self.args.remove_diffusion:
                # Get labels for efficient sampling from each class
                ood_samples = Subset(train_dataset, indices_to_be_augmented)
                batch_labels = self.get_batch_labels(
                    ood_samples, range(len(ood_samples))
                )

                # Log class distribution
                unique_labels, counts = torch.unique(batch_labels, return_counts=True)
                print("\nOOD samples class distribution:")
                for label, count in zip(
                    unique_labels.cpu().tolist(), counts.cpu().tolist()
                ):
                    print(f"Class {label}: {count} samples")

                # Add samples back to dataset
                self.added_indices.update(indices_to_be_augmented)
                self.datamodule.add_samples_by_index(indices_to_be_augmented)
                print(
                    f"Added {len(indices_to_be_augmented)} samples back to the training set"
                )
            else:
                ood_samples = Subset(train_dataset, indices_to_be_augmented)
                ood_labels = self.get_batch_labels(ood_samples, range(len(ood_samples)))

                expected_new_images = (
                    len(ood_samples) * self.args.num_generations_per_ood_sample
                )
                print(
                    f"\nGenerating {self.args.num_generations_per_ood_sample} images for each of {len(ood_samples)} OOD samples"
                )
                print(f"Expected total new images: {expected_new_images}")

                diffusion_pipe = self.initialize_model(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.generate_new_data(
                    ood_samples,
                    pipe=diffusion_pipe,
                    save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
                )

                # Add generated images to dataset
                self.datamodule.train_dataset.dataset.add_generated_images(
                    cycle_idx,
                    len(ood_samples) * self.args.num_generations_per_ood_sample,
                    ood_labels.repeat_interleave(
                        self.args.num_generations_per_ood_sample
                    ).tolist(),
                )

                # Update dataset with new images
                self.datamodule.train_dataset = Subset(
                    self.datamodule.train_dataset.dataset,
                    list(range(len(self.datamodule.train_dataset.dataset))),
                )

                # Clean up GPU memory after diffusion
                if torch.cuda.is_available():
                    del diffusion_pipe
                    torch.cuda.empty_cache()

            # Reset transforms
            self.datamodule.train_dataset.dataset.transform = ssl_transform

            # Verify dataset size change
            final_size = len(self.datamodule.train_dataset)
            if self.args.remove_diffusion:
                expected_increase = len(indices_to_be_augmented)
            else:
                expected_increase = (
                    len(ood_samples) * self.args.num_generations_per_ood_sample
                )

            print(f"\nDataset size verification:")
            print(f"Initial size: {initial_size}")
            print(f"Final size: {final_size}")
            print(f"Expected increase: {expected_increase}")
            print(f"Actual increase: {final_size - initial_size}")

            if final_size != initial_size + expected_increase:
                raise AssertionError(
                    f"Dataset size mismatch. Expected {initial_size + expected_increase}, "
                    f"got {final_size}"
                )

            print(
                f"✓ Dataset size verified: Successfully added {expected_increase} images"
            )

        finally:
            self._cleanup_cycle_resources(trainer)

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

    def get_random_indices(self, dataset) -> list:
        """Get random indices for augmentation"""
        return torch.randperm(len(dataset))[
            : self.args.num_ood_samples * self.args.num_generations_per_ood_sample
        ].tolist()

    def get_ood_indices(self, dataset, cycle_idx) -> list:
        """Get indices of OOD samples using feature-based detection"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_method.to(device)
        self.ssl_method.model.to(dtype=self.ssl_method.dtype)

        ood = OOD(
            args=self.args,
            dataset=dataset,  # Now takes single dataset
            feature_extractor=self.ssl_method.model.extract_features,
            cycle_idx=cycle_idx,
            device=self.ssl_method.device,
            dtype=self.ssl_method.dtype,
        )

        ood_indices, _ = ood.ood()
        return ood_indices

    def save_class_dist(self, cycle_idx: int) -> None:
        """Save class distribution for current cycle using GPU acceleration"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = self.datamodule.train_dataset.dataset
        num_classes = dataset.num_classes
        class_counts = torch.zeros(num_classes, device=device)

        # Process in batches
        loader = DataLoader(
            self.datamodule.train_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Count labels in batches
        for _, labels in tqdm(
            loader, desc=f"Counting class distribution for cycle {cycle_idx}"
        ):
            labels = labels.to(device)
            counts = torch.bincount(labels, minlength=num_classes)
            class_counts += counts

        # Save distribution
        save_path = f"{os.environ['BASE_CACHE_DIR']}/class_distributions/{self.checkpoint_filename}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(class_counts.cpu(), f"{save_path}/dist_cycle_{cycle_idx}.pt")

        # Log distribution info
        print(f"\nCycle {cycle_idx} distribution:")
        print(f"Total samples: {class_counts.sum().item()}")
        print(f"Non-zero classes: {(class_counts > 0).sum().item()}")
        print(f"Mean samples per class: {class_counts.mean().item():.2f}")
        print(f"Std samples per class: {class_counts.std().item():.2f}")

    def pretrain_imbalanced(self) -> None:
        """Run the main training loop with OOD detection and augmentation"""
        visualization_dir = (
            f"visualizations/class_distributions/{self.checkpoint_filename}"
        )
        os.makedirs(visualization_dir, exist_ok=True)

        for cycle_idx in range(self.max_cycles):
            print(f"Run {self.run_idx + 1}/{self.args.num_runs}")
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")

            # Train for one cycle
            self.pretrain_cycle(cycle_idx)

            # Save and visualize class distribution
            if self.save_class_distribution:
                self.save_class_dist(cycle_idx + 1)

    def finetune(self) -> dict:
        """Run finetuning on benchmark datasets"""
        benchmarks = FinetuningBenchmarks.get_benchmarks(
            self.args.finetuning_benchmarks
        )
        results = {}

        self.trainer_args.pop("callbacks")
        torch.multiprocessing.set_sharing_strategy("file_system")

        for benchmark in benchmarks:
            print(f"\n -- Finetuning benchmark: {benchmark.__name__} --\n")

            transform = transforms.Compose(
                [
                    transforms.Resize((self.args.crop_size, self.args.crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

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

            self.trainer_args["callbacks"] = (
                [early_stop_callback] if "KNN" not in benchmark.__name__ else []
            )
            self.trainer_args["max_time"] = {"minutes": 25}
            self.trainer_args["accumulate_grad_batches"] = 1

            if torch.cuda.is_available():
                strategy = DeepSpeedStrategy(
                    config={
                        "train_batch_size": self.args.train_batch_size
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
                self.trainer_args["strategy"] = strategy
                self.trainer_args.pop("accelerator", None)

            trainer = L.Trainer(**self.trainer_args)
            trainer.fit(model=finetuner)
            finetuning_results = trainer.test(model=finetuner)[0]
            results.update(finetuning_results)

            # Clean up GPU memory after each benchmark
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def initialize_model(self, device: str) -> StableUnCLIPImg2ImgPipeline:
        """Initialize the diffusion model"""
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
        return pipe.to(device)

    def _cleanup_cycle_resources(self, trainer: L.Trainer) -> None:
        """Clean up resources after each training cycle"""
        self.datamodule.set_dataloaders_none()

        if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "cleanup"):
            trainer.strategy.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_new_data(self, ood_samples, pipe, save_subfolder) -> None:
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
            range(0, len(ood_samples), self.args.sd_batch_size),
            desc="Generating New Data...",
        ):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            # Get and denormalize batch
            batch = [
                ood_samples[i + b_start][0]
                for i in range(min(len(ood_samples) - b_start, self.args.sd_batch_size))
            ]
            batch = [denorm(img) for img in batch]
            # Generate images
            v_imgs = pipe(
                batch, num_images_per_prompt=self.args.num_generations_per_ood_sample
            ).images
            # Save batch
            image_storage.save_batch(v_imgs, cycle_idx, k)
            k += len(v_imgs)
            sys.stdout = old_stdout
