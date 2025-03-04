import sys
from torchvision.utils import save_image
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
from diffusers import (
    StableDiffusionImageVariationPipeline,
    FluxPriorReduxPipeline,
    FluxPipeline,
)
from torchvision import transforms
import copy
import matplotlib.pyplot as plt

import os
import pickle

from experiment.dataset.ImageStorage import ImageStorage
from experiment.utils.get_num_workers import get_num_workers


class FluxAugmentor:
    def __init__(self):
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def augment(self, images, num_generations_per_image=1):
        pipe_prior_output = self.pipe_prior_redux(images)

        return self.pipe(
            num_images_per_prompt=num_generations_per_image,
            num_inference_steps=20,
            **pipe_prior_output,
        ).images


class StableDiffusionAugmentor:
    def __init__(self):
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
        ).to("cuda")

    def augment(self, images, num_generations_per_image=1):
        return self.pipe(
            images,
            num_inference_steps=20,
            num_images_per_prompt=num_generations_per_image,
        ).images


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

        self.num_workers = min(6, get_num_workers() // 2)

        if self.save_class_distribution:
            self.save_class_dist(0)

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

    def get_sorted_classes_distribution(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        class_counts = torch.zeros(dataset.num_classes)

        for _, labels in tqdm(dataloader, desc="Counting class distribution"):
            counts = torch.bincount(labels, minlength=dataset.num_classes)
            class_counts += counts

        return class_counts

    def pretrain_cycle(self, cycle_idx) -> None:
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

            if cycle_idx >= self.max_cycles - 1:
                return

            if not self.args.ood_augmentation:
                print("OOD augmentation disabled, skipping generation")
                return

            ssl_transform = copy.deepcopy(
                self.datamodule.train_dataset.dataset.transform
            )
            self.datamodule.train_dataset.dataset.transform = self.transform

            # Log OOD selection method
            print(
                f"\nSelecting OOD samples using {'OOD detection' if self.args.use_ood else 'random selection'}"
            )

            ood_indices = (
                self.get_ood_indices(self.datamodule.train_dataset, cycle_idx)
                if self.args.use_ood
                else self.get_random_indices(self.datamodule.train_dataset)
            )

            ood_samples = [self.datamodule.train_dataset[i] for i in ood_indices]
            ood_labels = torch.tensor(
                [label for _, label in tqdm(ood_samples, desc="Getting labels")]
            )

            print(f"Selected {len(ood_indices)} samples for augmentation")

            if self.args.remove_diffusion:
                # Add samples back to dataset
                self.added_indices.update(ood_indices)
                self.datamodule.add_samples_by_index(ood_indices)
                print(f"Added {len(ood_indices)} samples back to the training set")
            else:
                expected_new_images = (
                    len(ood_samples) * self.args.num_generations_per_ood_sample
                )
                print(
                    f"\nGenerating {self.args.num_generations_per_ood_sample} images for each of {len(ood_samples)} OOD samples"
                )
                print(f"Expected total new images: {expected_new_images}")

                diffusion_pipe = (
                    StableDiffusionAugmentor()
                    if self.args.generation_model == "stable_diffusion"
                    else FluxAugmentor()
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

                # Get current subset indices
                current_indices = set(self.datamodule.train_dataset.indices)

                # Calculate new indices for added data
                base_length = len(self.datamodule.train_dataset.dataset) - (
                    len(ood_samples) * self.args.num_generations_per_ood_sample
                )
                new_indices = range(
                    base_length, len(self.datamodule.train_dataset.dataset)
                )

                # Combine old and new indices
                combined_indices = list(current_indices) + list(new_indices)

                # Update the subset with combined indices
                self.datamodule.train_dataset = Subset(
                    self.datamodule.train_dataset.dataset, combined_indices
                )

                # Clean up GPU memory after diffusion
                if torch.cuda.is_available():
                    del diffusion_pipe
                    torch.cuda.empty_cache()

            # Reset transforms
            self.datamodule.train_dataset.dataset.transform = ssl_transform

        finally:
            self._cleanup_cycle_resources()
            gc.collect()
            torch.cuda.empty_cache()

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
        return torch.randperm(len(dataset))[: self.args.num_ood_samples].tolist()

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

        ood_indices = ood.ood()

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
            num_workers=0,
            pin_memory=False,
        )

        # Count labels in batches
        for _, labels in tqdm(
            loader, desc=f"Counting class distribution for cycle {cycle_idx}"
        ):
            labels = labels.to(device)
            counts = torch.bincount(labels, minlength=num_classes)
            class_counts += counts

        self.class_counts = class_counts

        # Create dictionary mapping class indices to class names
        class_names_dict = {}
        for idx in range(num_classes):
            class_names_dict[idx] = (
                self.datamodule.train_dataset.dataset.get_class_name(idx)
            )

        # Save distribution and class names
        save_path = f"{os.environ['BASE_CACHE_DIR']}/class_distributions/{self.checkpoint_filename}"
        os.makedirs(save_path, exist_ok=True)

        # Save counts as tensor
        torch.save(class_counts.cpu(), f"{save_path}/dist_cycle_{cycle_idx}.pt")

        # Save class names dictionary
        torch.save(class_names_dict, f"{save_path}/class_names_cycle_{cycle_idx}.pt")

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

            import gc

            gc.collect()
            torch.cuda.empty_cache()

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
            results.update(finetuning_results)

            # Clean up GPU memory after each benchmark
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _cleanup_cycle_resources(self) -> None:
        """Clean up resources after each training cycle"""
        if hasattr(self.datamodule, "_train_dataloader"):
            if hasattr(self.datamodule._train_dataloader, "_iterator"):
                self.datamodule._train_dataloader._iterator = None
        if hasattr(self.datamodule, "_val_dataloader"):
            if hasattr(self.datamodule._val_dataloader, "_iterator"):
                self.datamodule._val_dataloader._iterator = None
        if hasattr(self.datamodule, "_test_dataloader"):
            if hasattr(self.datamodule._test_dataloader, "_iterator"):
                self.datamodule._test_dataloader._iterator = None

        self.datamodule.set_dataloaders_none()

        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_new_data(self, ood_samples, pipe, save_subfolder) -> None:
        """
        Generate new data using the diffusion model.
        Processes images in batches according to sd_batch_size, potentially feeding the same image
        multiple times if num_generations_per_ood_sample is greater than sd_batch_size.
        """
        cycle_idx = int(save_subfolder.split("/")[-1])
        denorm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )

        total_images_saved = 0
        generations_per_batch = min(
            self.args.sd_batch_size, self.args.num_generations_per_ood_sample
        )

        # Create DataLoader with a batch size that accounts for multiple generations
        effective_batch_size = max(1, self.args.sd_batch_size // generations_per_batch)
        dataloader = DataLoader(
            ood_samples,
            batch_size=effective_batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        already_saved_sample_classes = set()

        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Generating New Data...")
        ):
            # Denormalize the batch
            batch = denorm(images)

            # Calculate how many passes we need for this batch
            remaining_generations = self.args.num_generations_per_ood_sample
            batch_images = []

            while remaining_generations > 0:
                # Calculate number of images to generate this pass
                current_generations = min(generations_per_batch, remaining_generations)

                # Generate images
                generated_images = pipe.augment(
                    batch,
                    num_generations_per_image=current_generations,
                )

                batch_images.extend(generated_images)
                remaining_generations -= current_generations

            if (
                self.args.save_class_distribution
                and labels[0] not in already_saved_sample_classes
            ):
                already_saved_sample_classes.add(labels[0])

                # Save generated image
                save_dir = (
                    f"{os.environ['BASE_CACHE_DIR']}/ood_samples/cycle_{cycle_idx}/"
                )
                os.makedirs(save_dir, exist_ok=True)
                filename_generated = (
                    self.datamodule.train_dataset.dataset.get_class_name(labels[0])
                )
                save_path_generated = f"{save_dir}/{filename_generated}_generated.png"
                batch_images[0].save(save_path_generated, "PNG")

                filename_original = f"{filename_generated}_original.png"
                save_path_original = f"{save_dir}/{filename_original}"
                save_image(images[0], save_path_original)

            # Save all generated images for this batch
            self.datamodule.train_dataset.dataset.image_storage.save_batch(
                batch_images, cycle_idx, total_images_saved
            )
            total_images_saved += len(batch_images)

        print(f"Total images generated and saved: {total_images_saved}")
