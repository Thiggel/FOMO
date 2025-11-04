import sys
from typing import Optional
from sklearn.manifold import TSNE
import numpy as np
import wandb
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import io
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Subset, random_split, Dataset, DataLoader
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
import re

try:
    from lightning.pytorch.strategies import DeepSpeedStrategy
except Exception:  # pragma: no cover - deepspeed optional
    DeepSpeedStrategy = None
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
    def __init__(self, device: Optional[str] = None, dtype: torch.dtype = torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=self.dtype,
        ).to(self.device)

    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        prompt: Optional[str] = None,
        num_steps: int = 6,
        guidance: float = 2.5,
    ):
        pipe_prior_output = self.pipe_prior_redux(image=images, prompt=prompt)

        return self.pipe(
            **pipe_prior_output,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_generations_per_image,
        ).images


class StableDiffusionAugmentor:
    def __init__(self):
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            safety_checker=None,
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
        self.n_epochs_per_cycle = args.n_epochs_per_cycle
        self.max_cycles = args.max_cycles
        self.completed_cycles = 0
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
        self.last_ood_results: Optional[dict] = None

        self.num_workers = min(6, get_num_workers() // 2)

        if self.datamodule is not None and self.args.logger:
            self._run_training_analysis(stage_label="start", cycle_reference=0)

    def _run_training_analysis(self, stage_label: str, cycle_reference: int) -> None:
        """Run configured analysis routines for a specific training stage."""

        if not self.args.logger or self.datamodule is None:
            return

        if self.args.log_tsne:
            self.visualize_embedding_space(cycle_reference, stage_label=stage_label)

        if self.args.log_class_dist:
            self.save_class_dist(cycle_reference, stage_label=stage_label)

        self._log_generated_samples_summary(stage_label)

    def _log_generated_samples_summary(self, stage_label: str) -> None:
        """Log a summary of generated samples for the current stage."""

        if not (self.args.logger and self.args.log_generated_samples):
            return

        wandb_logger = self.trainer_args.get("logger", None)
        if not wandb_logger or not hasattr(wandb_logger, "experiment"):
            return

        train_dataset = getattr(self.datamodule, "train_dataset", None)
        if train_dataset is None:
            return

        base_dataset = train_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset

        additional_counts = getattr(base_dataset, "additional_image_counts", None)

        if additional_counts is None:
            additional_counts = {}

        total_generated = sum(
            cycle_data.get("count", 0) for cycle_data in additional_counts.values()
        )

        summary_payload = {
            "stage": stage_label,
            "total_generated_images": int(total_generated),
            "cycles_with_generation": int(len(additional_counts)),
        }

        table = wandb.Table(columns=["cycle", "generated_images", "unique_classes"])

        for cycle_key, cycle_data in sorted(
            additional_counts.items(), key=lambda item: str(item[0])
        ):
            count = int(cycle_data.get("count", 0))
            labels = cycle_data.get("labels", []) or []
            unique_classes = len(set(labels))
            table.add_data(str(cycle_key), count, int(unique_classes))

        wandb_logger.experiment.log(
            {
                f"generated_samples/summary_{stage_label}": summary_payload,
                f"generated_samples/table_{stage_label}": table,
                "analysis_stage": stage_label,
            }
        )

    @staticmethod
    def _sanitize_artifact_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
        return sanitized or "artifact"

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

    def get_outliers(self, cycle_idx):
        if self.args.sample_selection == "ood":
            print("Using OOD detection for sample selection")
            ood_indices = self.get_ood_indices(self.datamodule.train_dataset, cycle_idx)
        elif self.args.sample_selection == "oracle":
            print("Using oracle indices for sample selection")
            ood_indices = self.get_oracle_indices()
        else:
            print("Using random selection for sample selection")
            ood_indices = self.get_random_indices(self.datamodule.train_dataset)

        return ood_indices

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
                if self.args.use_deepspeed and DeepSpeedStrategy is not None:
                    strategy = DeepSpeedStrategy(
                        config={
                            "train_batch_size": self.args.train_batch_size
                            * self.args.grad_acc_steps
                            * torch.cuda.device_count(),
                            "bf16": {"enabled": False},
                            "zero_optimization": {
                                "stage": 2,
                            },
                            "zero_allow_untested_optimizer": True,
                        },
                    )
                    cycle_trainer_args["strategy"] = strategy
                else:
                    cycle_trainer_args.pop("strategy", None)

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

            ood_indices = self.get_outliers(cycle_idx)

            ood_samples = [self.datamodule.train_dataset[i] for i in ood_indices]
            ood_labels = torch.tensor(
                [label for _, label in tqdm(ood_samples, desc="Getting labels")]
            )

            print(f"Selected {len(ood_indices)} samples for augmentation")

            if (
                self.args.logger
                and (wandb_logger := self.trainer_args.get("logger", None))
                and hasattr(wandb_logger, "experiment")
            ):
                cycle_log_idx = cycle_idx + 1
                self._log_ood_class_distribution(
                    cycle_log_idx, ood_labels, wandb_logger
                )
                self._log_ood_partition_summary(cycle_log_idx, wandb_logger)
                self._log_ood_distance_cdf(cycle_log_idx, wandb_logger)

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

            if os.path.exists(self.checkpoint_callback.best_model_path):
                if self.args.use_deepspeed:
                    output_path = (
                        self.checkpoint_callback.best_model_path
                        + "_fp32.pt".replace(":", "_").replace(" ", "_")
                    )

                    convert_zero_checkpoint_to_fp32_state_dict(
                        self.checkpoint_callback.best_model_path,
                        output_path,
                    )

                    self.ssl_method.load_state_dict(torch.load(output_path)["state_dict"])
                else:
                    checkpoint = torch.load(self.checkpoint_callback.best_model_path)
                    state_dict = (
                        checkpoint["state_dict"]
                        if "state_dict" in checkpoint
                        else checkpoint
                    )
                    self.ssl_method.load_state_dict(state_dict)

        if self.datamodule is not None and self.args.logger:
            final_cycle_reference = (
                self.completed_cycles
                if self.completed_cycles
                else (self.max_cycles if self.max_cycles else 0)
            )
            self._run_training_analysis(
                stage_label="end", cycle_reference=final_cycle_reference
            )

        return self.finetune() if self.args.finetune else {}

    def get_random_indices(self, dataset) -> list:
        """Get random indices for augmentation"""
        return torch.randperm(len(dataset))[: self.args.num_ood_samples].tolist()

    def get_oracle_indices(self) -> list:
        inverse_distribution, labels = (
            self.datamodule.train_dataset.dataset._get_inverse_distribution()
        )

        dist_sorted_by_imbalance = torch.argsort(inverse_distribution, descending=False)
        labels_sorted_by_imbalance = labels[dist_sorted_by_imbalance]

        oracle_ood_indices = labels_sorted_by_imbalance[: self.args.num_ood_samples]

        print(oracle_ood_indices.shape)
        print(f"Oracle indices: {oracle_ood_indices.tolist()}")
        return oracle_ood_indices.tolist()

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
        self.last_ood_results = ood.last_results

        return ood_indices

    def collect_embeddings(
        self, tsne_max_samples: int = 10000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        old_transform = self.datamodule.train_dataset.dataset.transform

        self.datamodule.train_dataset.dataset.transform = transforms.Compose(
            [
                transforms.Resize((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = self.datamodule.train_dataset
        if tsne_max_samples is not None and tsne_max_samples < len(dataset):
            subset_indices = torch.randperm(len(dataset))[:tsne_max_samples].tolist()
            dataset = Subset(dataset, subset_indices)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
        )

        embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
                images = images.to(
                    device=self.ssl_method.device, dtype=self.ssl_method.dtype
                )
                labels = labels.to(device=self.ssl_method.device)

                batch_embeddings = self.ssl_method.model.extract_features(images).cpu()
                embeddings.append(batch_embeddings)
                all_labels.append(labels)

        self.datamodule.train_dataset.dataset.transform = old_transform

        return torch.cat(embeddings, dim=0), torch.cat(all_labels, dim=0)

    def apply_tsne(self, embeddings, labels) -> np.ndarray:
        tsne = TSNE(
            n_components=2,
            random_state=0,
            perplexity=40,
            learning_rate="auto",
            init="pca",
        )
        tsne_embeddings = tsne.fit_transform(embeddings.numpy())

        return tsne_embeddings

    def generate_colors(self, n):
        """Generate n visually distinct colors using HSV color space"""
        import colorsys
        import numpy as np

        colors = []
        # Use golden ratio to space hues around the color wheel
        golden_ratio_conjugate = 0.618033988749895
        h = 0.1  # Starting hue

        # Generate colors with varying hue, saturation, and value
        for i in range(n):
            # Primary variation is hue
            h = (h + golden_ratio_conjugate) % 1.0
            # Secondary variations in saturation and value
            s = 0.5 + 0.5 * ((i % 7) / 6.0)  # 7 saturation levels
            v = 0.9 - 0.4 * ((i % 5) / 4.0)  # 5 value levels

            rgb = colorsys.hsv_to_rgb(h, s, v)
            colors.append(rgb)

        return colors

    def plot_tsne(
        self,
        tsne_embeddings,
        labels,
        class_names=None,
        fig_size=(12, 10),
        ood_mask=None,
    ):
        plt.figure(figsize=fig_size)

        labels_np = labels.cpu().numpy()
        ood_mask_np = ood_mask.cpu().numpy() if ood_mask is not None else None

        unique_classes = np.unique(labels_np)
        colors = self.generate_colors(len(unique_classes))

        for cls in unique_classes:
            mask = (labels_np == cls) & (
                ~ood_mask_np if ood_mask_np is not None else True
            )
            plt.scatter(
                tsne_embeddings[mask, 0],
                tsne_embeddings[mask, 1],
                s=3,
                color=colors[cls],
                label=class_names[cls] if class_names else cls,
                alpha=0.5,
            )

        for cls in unique_classes:
            mask = (labels_np == cls) & (
                ood_mask_np if ood_mask_np is not None else True
            )
            plt.scatter(
                tsne_embeddings[mask, 0],
                tsne_embeddings[mask, 1],
                s=10,
                color=colors[cls],
                label=class_names[cls] if class_names else cls,
                edgecolors="black",
                alpha=1.0,
            )

        legend = plt.legend(
            bbox_to_anchor=(1.05, 1),  # Position right of the plot
            loc="upper left",
            fontsize="xx-small",  # Very small font
            markerscale=0.5,  # Smaller markers
            ncol=2,  # Use 2 columns to save vertical space
            frameon=True,  # Add a frame
            title="Classes",  # Add a title
            title_fontsize="small",
        )

        for handle in legend.legend_handles:
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([10])

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        return plt.gcf()

    def visualize_embedding_space(
        self, cycle_idx: int, stage_label: Optional[str] = None
    ) -> None:

        old_transform = self.datamodule.train_dataset.dataset.transform
        self.datamodule.train_dataset.dataset.transform = transforms.Compose(
            [
                transforms.Resize((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        ood_indices = self.get_outliers(cycle_idx)

        self.datamodule.train_dataset.dataset.transform = old_transform
        embeddings, labels = self.collect_embeddings(self.args.tsne_max_samples)

        print("Computing t-SNE embeddings...")
        tsne_embeddings = self.apply_tsne(embeddings, labels)

        class_names = {
            idx: self.datamodule.train_dataset.dataset.get_class_name(idx)
            for idx in range(self.datamodule.train_dataset.dataset.num_classes)
        }

        ood_mask = torch.zeros(len(labels), dtype=torch.bool)
        ood_indices = [idx for idx in ood_indices if 0 <= idx < len(labels)]
        ood_mask[ood_indices] = True

        fig = self.plot_tsne(tsne_embeddings, labels, class_names, ood_mask=ood_mask)

        vis_dir = f"{os.environ['BASE_CACHE_DIR']}/visualizations/tsne/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)
        cycle_label = stage_label if stage_label is not None else cycle_idx
        png_path = f"{vis_dir}/tsne_cycle_{cycle_label}.png"
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # 5. Log to Wandb
        if (
            self.args.logger
            and self.args.log_tsne
            and hasattr(self.trainer_args.get("logger", None), "experiment")
        ):
            wandb_logger = self.trainer_args["logger"]
            wandb_logger.experiment.log(
                {
                    f"tsne/cycle_{cycle_label}": wandb.Image(png_path),
                    "cycle": cycle_label,
                    "analysis_stage": stage_label or "cycle",
                }
            )

        del embeddings, tsne_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_class_dist(
        self, cycle_idx: int, stage_label: Optional[str] = None
    ) -> None:
        """Save class distribution for current cycle using GPU acceleration and log to Wandb"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = self.datamodule.train_dataset.dataset
        num_classes = dataset.num_classes
        class_counts = torch.zeros(
            num_classes, device=device, dtype=torch.long
        )

        # Process in batches
        loader = DataLoader(
            self.datamodule.train_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=0,
            pin_memory=False,
        )

        # Count labels in batches
        for _, labels in tqdm(
            loader,
            desc=f"Counting class distribution for cycle {stage_label if stage_label is not None else cycle_idx}"
        ):
            labels = labels.to(device)
            counts = torch.bincount(labels, minlength=num_classes)
            class_counts += counts

        class_counts_cpu = class_counts.cpu()
        self.class_counts = class_counts_cpu

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
        cycle_label = stage_label if stage_label is not None else cycle_idx
        torch.save(class_counts.cpu(), f"{save_path}/dist_cycle_{cycle_label}.pt")

        # Save class names dictionary
        torch.save(class_names_dict, f"{save_path}/class_names_cycle_{cycle_label}.pt")

        # Log distribution info
        print(f"\nCycle {cycle_label} distribution:")
        counts_float = class_counts_cpu.float()
        print(f"Total samples: {counts_float.sum().item()}")
        print(f"Non-zero classes: {(class_counts_cpu > 0).sum().item()}")
        print(f"Mean samples per class: {counts_float.mean().item():.2f}")
        print(f"Std samples per class: {counts_float.std().item():.2f}")

        # Create visualization and log to Wandb
        if (
            self.args.logger
            and self.args.log_class_dist
            and hasattr(self.trainer_args.get("logger", None), "experiment")
        ):
            try:
                # Create plot using matplotlib
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt

                # Convert to numpy for plotting
                original_counts, generated_counts = (
                    self._compute_original_generated_counts(class_counts_cpu)
                )
                counts_np = class_counts_cpu.numpy()
                original_np = original_counts.numpy()
                generated_np = generated_counts.numpy()

                # Create distribution plot
                fig, ax = plt.subplots(figsize=(12, 6))
                indices = np.arange(num_classes)
                ax.bar(
                    indices,
                    original_np,
                    label="Original",
                    color="tab:blue",
                )
                ax.bar(
                    indices,
                    generated_np,
                    bottom=original_np,
                    label="Generated",
                    color="tab:orange",
                )
                ax.set_xlabel("Class Index")
                ax.set_ylabel("Number of Samples")
                ax.set_title(f"Class Distribution - Cycle {cycle_label}")
                ax.legend()

                # Save as PNG instead of PDF
                vis_dir = (
                    f"visualizations/class_distributions/{self.checkpoint_filename}"
                )
                os.makedirs(vis_dir, exist_ok=True)
                png_path = f"{vis_dir}/class_dist_cycle_{cycle_label}.png"
                pdf_path = f"{vis_dir}/class_dist_cycle_{cycle_label}.pdf"
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close(fig)

                # Log PNG to Wandb
                wandb_logger = self.trainer_args["logger"]
                wandb_logger.experiment.log(
                    {
                        f"class_distribution/cycle_{cycle_label}": wandb.Image(png_path),
                        f"class_distribution_pdf/cycle_{cycle_label}": wandb.File(pdf_path),
                        "cycle": cycle_label,
                        "analysis_stage": stage_label or "cycle",
                        "class_distribution_stats": {
                            "total_samples": counts_float.sum().item(),
                            "non_zero_classes": (class_counts_cpu > 0).sum().item(),
                            "mean_samples_per_class": counts_float.mean().item(),
                            "std_samples_per_class": counts_float.std().item(),
                        },
                    }
                )

                # Optionally, create a more detailed visualization for classes with samples
                if (
                    class_counts_cpu > 0
                ).sum().item() < 50:  # Only if there are fewer than 50 classes with samples
                    # Get indices of classes with samples
                    non_zero_indices = (
                        torch.nonzero(class_counts_cpu).squeeze().cpu().numpy()
                    )

                    # Create bar plot with class names
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(
                        non_zero_indices,
                        counts_np[non_zero_indices],
                    )

                    # Add class names as labels
                    class_names = [
                        class_names_dict.get(idx, f"Class {idx}")
                        for idx in non_zero_indices
                    ]
                    plt.xticks(non_zero_indices, class_names, rotation=90)
                    plt.xlabel("Class Name")
                    plt.ylabel("Number of Samples")
                    plt.title(
                        f"Class Distribution (Non-zero Classes) - Cycle {cycle_label}"
                    )
                    plt.tight_layout()

                    # Save detailed plot
                    detailed_png_path = (
                        f"{vis_dir}/class_dist_detailed_cycle_{cycle_label}.png"
                    )
                    plt.savefig(detailed_png_path, dpi=100, bbox_inches="tight")
                    plt.close()

                    # Log detailed PNG to Wandb
                    wandb_logger.experiment.log(
                        {
                            f"class_distribution_detailed/cycle_{cycle_label}": wandb.Image(
                                detailed_png_path
                            ),
                            "cycle": cycle_label,
                            "analysis_stage": stage_label or "cycle",
                        }
                    )

            except Exception as e:
                print(f"Warning: Failed to log class distribution to wandb: {str(e)}")

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
            if self.args.logger and self.args.log_tsne:
                self.visualize_embedding_space(cycle_idx + 1)
            if self.args.logger and self.args.log_class_dist:
                self.save_class_dist(cycle_idx + 1)

            self.completed_cycles = cycle_idx + 1

    def finetune(self) -> dict:
        """Run finetuning on benchmark datasets"""
        benchmarks = FinetuningBenchmarks.benchmarks
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
                lr=self.args.ssl.lr,
                transform=transform,
                crop_size=self.args.crop_size,
            )

            self.trainer_args["max_epochs"] = finetuner.max_epochs

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=20,
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
                if (
                    self.args.use_deepspeed
                    and DeepSpeedStrategy is not None
                    and getattr(finetuner, "use_deepspeed", True)
                ):
                    strategy = DeepSpeedStrategy(
                        config={
                            "train_batch_size": 64 * torch.cuda.device_count(),
                            "train_micro_batch_size_per_gpu": 64,
                            "zero_optimization": {"stage": 1},
                            "zero_allow_untested_optimizer": True,
                        },
                    )
                    self.trainer_args["strategy"] = strategy
                    self.trainer_args.pop("accelerator", None)
                else:
                    self.trainer_args.pop("strategy", None)
                    self.trainer_args["accelerator"] = "cuda"
                    self.trainer_args["devices"] = "auto"

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

    def _prepare_wandb_image_pair(self, original, generated, denorm, image_resize):
        """Convert original and generated images to small PIL images for Wandb logging."""
        if torch.is_tensor(original):
            original_tensor = original.detach().cpu()
            if original_tensor.min() < 0:
                original_tensor = denorm(original_tensor)
            original_tensor = image_resize(original_tensor)
            original_img = transforms.ToPILImage()(original_tensor)
        else:
            original_img = image_resize(original)

        if torch.is_tensor(generated):
            generated_tensor = generated.detach().cpu()
            generated_tensor = image_resize(generated_tensor)
            generated_img = transforms.ToPILImage()(generated_tensor)
        else:
            generated_img = image_resize(generated)

        return original_img, generated_img

    def _get_label_for_dataset_index(self, dataset, index: int) -> int:
        """Resolve the label for a dataset index, handling Subset nesting and generated data."""
        if isinstance(dataset, Subset):
            base_index = dataset.indices[index]
            return self._get_label_for_dataset_index(dataset.dataset, base_index)

        if hasattr(dataset, "indices") and hasattr(dataset, "label_tensor"):
            if index < len(dataset.indices):
                dataset_index = dataset.indices[index]
                return int(dataset.label_tensor[dataset_index].item())

            _, _, label = dataset._get_additional_image_info(index)
            return int(label)

        _, label = dataset[index]
        if torch.is_tensor(label):
            return int(label.item())

        return int(label)

    def _get_top_class_names(self, labels: list[int], top_k: int = 3) -> list[str]:
        """Return the most common class names for the provided labels."""
        if not labels:
            return []

        dataset = self.datamodule.train_dataset.dataset
        label_tensor = torch.tensor(labels, dtype=torch.long)
        counts = torch.bincount(label_tensor, minlength=dataset.num_classes)
        nonzero_indices = torch.nonzero(counts, as_tuple=False).squeeze()

        if nonzero_indices.numel() == 0:
            return []

        nonzero_counts = counts[nonzero_indices]
        sorted_counts, order = torch.sort(nonzero_counts, descending=True)
        top_indices = nonzero_indices[order][:top_k]

        return [dataset.get_class_name(idx.item()) for idx in top_indices]

    def _compute_original_generated_counts(
        self, total_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split total counts into original and generated components."""

        train_subset = self.datamodule.train_dataset
        base_dataset = train_subset.dataset

        if not hasattr(train_subset, "indices"):
            return total_counts.clone(), torch.zeros_like(total_counts)

        subset_indices_tensor = torch.tensor(train_subset.indices, dtype=torch.long)
        original_mask = subset_indices_tensor < len(base_dataset.indices)

        if original_mask.any():
            base_indices_tensor = torch.tensor(base_dataset.indices, dtype=torch.long)
            original_dataset_indices = base_indices_tensor[
                subset_indices_tensor[original_mask]
            ]
            original_labels = base_dataset.label_tensor[original_dataset_indices]
            original_counts = torch.bincount(
                original_labels.cpu(), minlength=total_counts.numel()
            )
        else:
            original_counts = torch.zeros_like(total_counts, dtype=torch.long)

        original_counts = original_counts.to(
            device=total_counts.device, dtype=torch.long
        )
        generated_counts = (
            total_counts.to(torch.long) - original_counts
        ).clamp(min=0)

        return original_counts, generated_counts.to(device=total_counts.device)

    def _log_ood_class_distribution(
        self,
        cycle_idx: int,
        ood_labels: torch.Tensor,
        wandb_logger,
    ) -> None:
        if ood_labels.numel() == 0:
            return

        dataset = self.datamodule.train_dataset.dataset
        counts = torch.bincount(
            ood_labels.cpu(), minlength=dataset.num_classes
        )
        nonzero_mask = counts > 0
        if nonzero_mask.sum().item() == 0:
            return

        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze()
        nonzero_counts = counts[nonzero_indices]
        sorted_counts, order = torch.sort(nonzero_counts, descending=True)
        sorted_indices = nonzero_indices[order]
        top_k = min(20, sorted_indices.numel())
        top_indices = sorted_indices[:top_k]
        top_counts = sorted_counts[:top_k]
        class_names = [
            dataset.get_class_name(idx.item()) for idx in top_indices
        ]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = (
            f"visualizations/ood_class_distributions/{self.checkpoint_filename}"
        )
        os.makedirs(vis_dir, exist_ok=True)
        png_path = f"{vis_dir}/ood_class_dist_cycle_{cycle_idx}.png"
        pdf_path = f"{vis_dir}/ood_class_dist_cycle_{cycle_idx}.pdf"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(class_names)), top_counts.numpy())
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylabel("Number of OOD samples")
        ax.set_title(f"Top OOD Classes - Cycle {cycle_idx}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        wandb_logger.experiment.log(
            {
                f"ood_class_distribution/cycle_{cycle_idx}": wandb.Image(png_path),
                "cycle": cycle_idx,
            }
        )

        if hasattr(wandb_logger.experiment, "log_artifact"):
            artifact_name = self._sanitize_artifact_name(
                f"{self.checkpoint_filename}_cycle_{cycle_idx}_ood_class_dist_pdf"
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="visualization",
                metadata={"cycle": cycle_idx},
            )
            artifact.add_file(pdf_path, name=os.path.basename(pdf_path))
            wandb_logger.experiment.log_artifact(artifact)

    def _log_ood_partition_summary(self, cycle_idx: int, wandb_logger) -> None:
        if not self.last_ood_results:
            return

        distances = self.last_ood_results.get("distances")
        dataset_indices = self.last_ood_results.get("dataset_indices")
        sorted_indices = self.last_ood_results.get("sorted_indices")

        if distances is None or dataset_indices is None or sorted_indices is None:
            return

        num_samples = len(distances)
        if num_samples == 0:
            return

        percentile_ranges = [(0.95, 1.0), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85)]
        summary = {}
        table = wandb.Table(columns=["OOD percentile", "Top classes"])

        for lower, upper in percentile_ranges:
            start_idx = int(np.floor((1 - upper) * num_samples))
            end_idx = int(np.ceil((1 - lower) * num_samples))

            if end_idx <= start_idx:
                top_class_names = []
            else:
                positions = sorted_indices[start_idx:end_idx]
                labels = [
                    self._get_label_for_dataset_index(
                        self.datamodule.train_dataset, dataset_indices[pos]
                    )
                    for pos in positions
                ]
                top_class_names = self._get_top_class_names(labels)

            range_key = f"{int(lower * 100)}-{int(upper * 100)}%"
            padded_top = (top_class_names + ["N/A"] * 3)[:3]
            summary[range_key] = padded_top
            table.add_data(range_key, ", ".join(padded_top))

        wandb_logger.experiment.log(
            {
                f"ood_partition_summary/cycle_{cycle_idx}": summary,
                f"ood_partition_summary_table/cycle_{cycle_idx}": table,
                "cycle": cycle_idx,
            }
        )

    def _log_ood_distance_cdf(self, cycle_idx: int, wandb_logger) -> None:
        if not self.last_ood_results:
            return

        distances = self.last_ood_results.get("distances")

        if distances is None or len(distances) == 0:
            return

        sorted_distances = np.sort(distances)
        num_points = sorted_distances.size

        if num_points == 0:
            return

        empirical_cdf = np.arange(1, num_points + 1) / num_points

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = f"visualizations/ood_distance_cdf/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)

        pdf_path = f"{vis_dir}/ood_distance_cdf_cycle_{cycle_idx}.pdf"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(sorted_distances, empirical_cdf, where="post", linewidth=2)
        ax.set_xlabel("k-NN distance")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"Empirical CDF of OOD Distances - Cycle {cycle_idx}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        wandb_logger.experiment.log(
            {f"ood_distance_cdf_pdf/cycle_{cycle_idx}": wandb.File(pdf_path), "cycle": cycle_idx}
        )

        if hasattr(wandb_logger.experiment, "log_artifact"):
            artifact_name = self._sanitize_artifact_name(
                f"{self.checkpoint_filename}_cycle_{cycle_idx}_ood_distance_cdf"
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="visualization",
                metadata={"cycle": cycle_idx},
            )
            artifact.add_file(pdf_path, name=os.path.basename(pdf_path))
            wandb_logger.experiment.log_artifact(artifact)

    def generate_new_data(self, ood_samples, pipe, save_subfolder) -> None:
        """
        Generate new data using the diffusion model and log examples to Wandb.
        For Flux model, process each image individually since it doesn't accept batches.
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

        # For Wandb logging
        has_wandb = self.args.logger and self.args.log_generated_samples
        image_resize = transforms.Resize((64, 64))  # Resize to 64x64 for Wandb
        wandb_logger = self.trainer_args.get("logger", None)
        log_every_n_classes = getattr(self.args, "log_every_n_classes", 1)
        max_wandb_pairs = getattr(self.args, "max_wandb_augmentation_pairs", 8)
        target_wandb_pairs = max(4, max_wandb_pairs)
        logged_classes = 0
        wandb_pairs = []

        total_images_saved = 0
        already_saved_sample_classes = set()
        dataset_obj = self.datamodule.train_dataset.dataset

        if self.args.generation_model == "flux":
            # Flux doesn't support batching, so process one image at a time
            for idx, (image, label) in enumerate(
                tqdm(ood_samples, desc="Generating New Data with Flux...")
            ):
                # Apply denormalization if the image is a tensor
                if torch.is_tensor(image):
                    image_pil = transforms.ToPILImage()(denorm(image))
                else:
                    image_pil = image  # If already PIL

                # Generate multiple augmentations for this image
                batch_images = pipe.augment(
                    [image_pil],  # Flux expects a list of PIL images
                    num_generations_per_image=self.args.num_generations_per_ood_sample,
                )

                label_int = int(label.item() if torch.is_tensor(label) else label)
                class_name = dataset_obj.get_class_name(label_int)

                orig_small = None
                gen_small = None
                if (
                    has_wandb
                    and wandb_logger
                    and hasattr(wandb_logger, "experiment")
                ):
                    orig_small, gen_small = self._prepare_wandb_image_pair(
                        image, batch_images[0], denorm, image_resize
                    )

                    if len(wandb_pairs) < target_wandb_pairs:
                        wandb_pairs.append(
                            (
                                class_name,
                                orig_small.copy()
                                if hasattr(orig_small, "copy")
                                else orig_small,
                                gen_small.copy()
                                if hasattr(gen_small, "copy")
                                else gen_small,
                            )
                        )

                    logged_classes += 1
                    if logged_classes % log_every_n_classes == 0:
                        wandb_logger.experiment.log(
                            {
                                f"cycle_{cycle_idx}/class_{class_name}/original": wandb.Image(
                                    orig_small
                                ),
                                f"cycle_{cycle_idx}/class_{class_name}/generated": wandb.Image(
                                    gen_small
                                ),
                                "cycle": cycle_idx,
                            }
                        )

                # Check if this class should be saved as an example (first time seeing this class)
                if (
                    self.args.save_class_distribution
                    and label not in already_saved_sample_classes
                ):
                    already_saved_sample_classes.add(label)

                    # Save generated image
                    save_dir = (
                        f"{os.environ['BASE_CACHE_DIR']}/ood_samples/cycle_{cycle_idx}/"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    # Get class name
                    # Save original and first generated image as examples
                    filename_generated = f"{class_name}_generated.png"
                    save_path_generated = f"{save_dir}/{filename_generated}"
                    batch_images[0].save(save_path_generated, "PNG")

                    filename_original = f"{class_name}_original.png"
                    save_path_original = f"{save_dir}/{filename_original}"

                    # Save original image
                    if torch.is_tensor(image):
                        save_image(image.cpu(), save_path_original)
                    else:
                        image.save(save_path_original, "PNG")

                # Save all generated images
                self.datamodule.train_dataset.dataset.image_storage.save_batch(
                    batch_images, cycle_idx, total_images_saved
                )
                total_images_saved += len(batch_images)

                # Log progress
                if idx % 10 == 0:
                    print(
                        f"Processed {idx}/{len(ood_samples)} images, generated {total_images_saved} augmentations"
                    )
        else:
            # Original code for Stable Diffusion (supports batching)
            generations_per_batch = min(
                self.args.sd_batch_size, self.args.num_generations_per_ood_sample
            )

            # Create DataLoader with a batch size that accounts for multiple generations
            effective_batch_size = max(
                1, self.args.sd_batch_size // generations_per_batch
            )
            dataloader = DataLoader(
                ood_samples,
                batch_size=effective_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )

            for batch_idx, (images, labels) in enumerate(
                tqdm(dataloader, desc="Generating New Data with Stable Diffusion...")
            ):
                # Denormalize the batch
                batch = denorm(images)

                # Calculate how many passes we need for this batch
                remaining_generations = self.args.num_generations_per_ood_sample
                batch_images = []

                while remaining_generations > 0:
                    # Calculate number of images to generate this pass
                    current_generations = min(
                        generations_per_batch, remaining_generations
                    )

                    # Generate images
                    generated_images = pipe.augment(
                        batch,
                        num_generations_per_image=current_generations,
                    )

                    batch_images.extend(generated_images)
                    remaining_generations -= current_generations

                label_int = int(labels[0].item())
                class_name = dataset_obj.get_class_name(label_int)

                orig_small = None
                gen_small = None
                if (
                    has_wandb
                    and wandb_logger
                    and hasattr(wandb_logger, "experiment")
                ):
                    orig_small, gen_small = self._prepare_wandb_image_pair(
                        images[0], batch_images[0], denorm, image_resize
                    )

                    if len(wandb_pairs) < target_wandb_pairs:
                        wandb_pairs.append(
                            (
                                class_name,
                                orig_small.copy()
                                if hasattr(orig_small, "copy")
                                else orig_small,
                                gen_small.copy()
                                if hasattr(gen_small, "copy")
                                else gen_small,
                            )
                        )

                    logged_classes += 1
                    if logged_classes % log_every_n_classes == 0:
                        wandb_logger.experiment.log(
                            {
                                f"cycle_{cycle_idx}/class_{class_name}/original": wandb.Image(
                                    orig_small
                                ),
                                f"cycle_{cycle_idx}/class_{class_name}/generated": wandb.Image(
                                    gen_small
                                ),
                                "cycle": cycle_idx,
                            }
                        )

                if (
                    self.args.save_class_distribution
                    and labels[0].item() not in already_saved_sample_classes
                ):
                    already_saved_sample_classes.add(labels[0].item())

                    # Save generated image locally
                    save_dir = (
                        f"{os.environ['BASE_CACHE_DIR']}/ood_samples/cycle_{cycle_idx}/"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    # Get class name for the current label
                    # Save original and generated images
                    filename_generated = f"{class_name}_generated.png"
                    save_path_generated = f"{save_dir}/{filename_generated}"
                    batch_images[0].save(save_path_generated, "PNG")

                    filename_original = f"{class_name}_original.png"
                    save_path_original = f"{save_dir}/{filename_original}"
                    save_image(images[0].cpu(), save_path_original)

                # Save all generated images for this batch
                self.datamodule.train_dataset.dataset.image_storage.save_batch(
                    batch_images, cycle_idx, total_images_saved
                )
                total_images_saved += len(batch_images)

        print(f"Total images generated and saved: {total_images_saved}")

        # Log a summary of how many classes were augmented
        if has_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
            if wandb_pairs:
                table = wandb.Table(columns=["class", "original", "generated"])
                for class_name, original_img, generated_img in wandb_pairs:
                    table.add_data(
                        class_name,
                        wandb.Image(
                            original_img,
                            caption=f"{class_name} - original",
                        ),
                        wandb.Image(
                            generated_img,
                            caption=f"{class_name} - augmented",
                        ),
                    )

                wandb_logger.experiment.log(
                    {
                        f"cycle_{cycle_idx}/augmentation_examples": table,
                        "cycle": cycle_idx,
                    }
                )

            wandb_logger.experiment.log(
                {
                    f"cycle_{cycle_idx}/augmented_classes": len(
                        already_saved_sample_classes
                    ),
                    "cycle": cycle_idx,
                }
            )
