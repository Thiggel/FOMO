import sys
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
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
            self.visualize_embedding_space(0)
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

            if self.args.sample_selection == "ood":
                print("Using OOD detection for sample selection")
                ood_indices = self.get_ood_indices(
                    self.datamodule.train_dataset, cycle_idx
                )
            elif self.args.sample_selection == "oracle":
                print("Using oracle indices for sample selection")
                ood_indices = self.get_oracle_indices()
            else:
                print("Using random selection for sample selection")
                ood_indices = self.get_random_indices(self.datamodule.train_dataset)

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

        return ood_indices

    def collect_embeddings(
        self, max_samples=10000
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

        dataloader = DataLoader(
            self.datamodule.train_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.num_workers,
        )

        embeddings = []
        all_labels = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(
                tqdm(dataloader, desc="Collecting embeddings")
            ):
                images = images.to(
                    device=self.ssl_method.device, dtype=self.ssl_method.dtype
                )
                labels = labels.to(device=self.ssl_method.device)

                batch_embeddings = self.ssl_method.model.extract_features(images).cpu()
                embeddings.append(batch_embeddings)
                all_labels.append(labels)

                sample_count += len(images)
                if sample_count >= max_samples:
                    break

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

    def fit_gmm(self, tsne_embeddings):
        n_components = 100

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            max_iter=100,
            n_init=5,
            random_state=0,
        )

        gmm.fit(tsne_embeddings)

        return gmm

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse based on a covariance matrix."""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the ellipse (scaled for 95% confidence interval)
        for nsig in [1, 2, 3]:
            ax.add_patch(
                Ellipse(
                    position,
                    width=nsig * width,
                    height=nsig * height,
                    angle=angle,
                    fill=False,
                    **kwargs,
                )
            )

    def plot_tsne(
        self,
        tsne_embeddings,
        labels,
        class_names=None,
        fig_size=(12, 10),
        ood_mask=None,
        gmm=None,
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

        # Plot GMM cluster contours
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, gmm.n_components))

        # Draw ellipses for each cluster
        for i, (weight, mean, covar) in enumerate(
            zip(gmm.weights_, gmm.means_, gmm.covariances_)
        ):
            # Skip clusters with very low weight (these might be noise)
            if weight < 0.01:
                continue

            # Draw ellipses at 1, 2, and 3 standard deviations
            draw_ellipse(
                mean,
                covar,
                ax=ax,
                edgecolor=cluster_colors[i],
                linewidth=2,
                linestyle="-",
                alpha=0.7,
                label=f"Cluster {i+1}" if i == 0 else "",
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
            handle.set_sizes([10])

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        return plt.gcf()

    def visualize_embedding_space(self, cycle_idx) -> None:

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

        ood_indices = self.get_ood_indices(self.datamodule.train_dataset, cycle_idx)
        self.datamodule.train_dataset.dataset.transform = old_transform
        embeddings, labels = self.collect_embeddings()

        print("Computing t-SNE embeddings...")
        tsne_embeddings = self.apply_tsne(embeddings, labels)

        class_names = {
            idx: self.datamodule.train_dataset.dataset.get_class_name(idx)
            for idx in range(self.datamodule.train_dataset.dataset.num_classes)
        }

        ood_mask = torch.zeros(len(labels), dtype=torch.bool)
        ood_indices = [idx for idx in ood_indices if 0 <= idx < len(labels)]
        ood_mask[ood_indices] = True

        gmm = self.fit_gmm(tsne_embeddings)

        fig = self.plot_tsne(
            tsne_embeddings, labels, class_names, ood_mask=ood_mask, gmm=gmm
        )

        vis_dir = f"{os.environ['BASE_CACHE_DIR']}/visualizations/tsne/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)
        png_path = f"{vis_dir}/tsne_cycle_{cycle_idx}.png"
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # 5. Log to Wandb
        if (
            self.args.logger
            and not self.args.test_mode
            and hasattr(self.trainer_args.get("logger", None), "experiment")
        ):
            wandb_logger = self.trainer_args["logger"]
            wandb_logger.experiment.log(
                {f"tsne/cycle_{cycle_idx}": wandb.Image(png_path), "cycle": cycle_idx}
            )

    def save_class_dist(self, cycle_idx: int) -> None:
        """Save class distribution for current cycle using GPU acceleration and log to Wandb"""
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

        # Create visualization and log to Wandb
        if (
            self.args.logger
            and not self.args.test_mode
            and hasattr(self.trainer_args.get("logger", None), "experiment")
        ):
            try:
                # Create plot using matplotlib
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt

                # Convert to numpy for plotting
                counts_np = class_counts.cpu().numpy()

                # Create distribution plot
                plt.figure(figsize=(12, 6))
                plt.bar(range(num_classes), counts_np)
                plt.xlabel("Class Index")
                plt.ylabel("Number of Samples")
                plt.title(f"Class Distribution - Cycle {cycle_idx}")

                # Save as PNG instead of PDF
                vis_dir = (
                    f"visualizations/class_distributions/{self.checkpoint_filename}"
                )
                os.makedirs(vis_dir, exist_ok=True)
                png_path = f"{vis_dir}/class_dist_cycle_{cycle_idx}.png"
                plt.savefig(png_path, dpi=100, bbox_inches="tight")
                plt.close()

                # Log PNG to Wandb
                wandb_logger = self.trainer_args["logger"]
                wandb_logger.experiment.log(
                    {
                        f"class_distribution/cycle_{cycle_idx}": wandb.Image(png_path),
                        "cycle": cycle_idx,
                        "class_distribution_stats": {
                            "total_samples": class_counts.sum().item(),
                            "non_zero_classes": (class_counts > 0).sum().item(),
                            "mean_samples_per_class": class_counts.mean().item(),
                            "std_samples_per_class": class_counts.std().item(),
                        },
                    }
                )

                # Optionally, create a more detailed visualization for classes with samples
                if (
                    class_counts > 0
                ).sum().item() < 50:  # Only if there are fewer than 50 classes with samples
                    # Get indices of classes with samples
                    non_zero_indices = (
                        torch.nonzero(class_counts).squeeze().cpu().numpy()
                    )

                    # Create bar plot with class names
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(non_zero_indices, counts_np[non_zero_indices])

                    # Add class names as labels
                    class_names = [
                        class_names_dict.get(idx, f"Class {idx}")
                        for idx in non_zero_indices
                    ]
                    plt.xticks(non_zero_indices, class_names, rotation=90)
                    plt.xlabel("Class Name")
                    plt.ylabel("Number of Samples")
                    plt.title(
                        f"Class Distribution (Non-zero Classes) - Cycle {cycle_idx}"
                    )
                    plt.tight_layout()

                    # Save detailed plot
                    detailed_png_path = (
                        f"{vis_dir}/class_dist_detailed_cycle_{cycle_idx}.png"
                    )
                    plt.savefig(detailed_png_path, dpi=100, bbox_inches="tight")
                    plt.close()

                    # Log detailed PNG to Wandb
                    wandb_logger.experiment.log(
                        {
                            f"class_distribution_detailed/cycle_{cycle_idx}": wandb.Image(
                                detailed_png_path
                            ),
                            "cycle": cycle_idx,
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
            if self.save_class_distribution:
                self.visualize_embedding_space(cycle_idx + 1)
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
        has_wandb = self.args.logger and not self.args.test_mode
        image_resize = transforms.Resize((64, 64))  # Resize to 64x64 for Wandb

        # Store class-wise examples for Wandb
        wandb_originals = {}
        wandb_generated = {}
        wandb_class_names = {}

        total_images_saved = 0
        already_saved_sample_classes = set()

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
                    class_name = self.datamodule.train_dataset.dataset.get_class_name(
                        label
                    )

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

                    # For Wandb logging
                    if has_wandb:
                        # Prepare original image for wandb
                        if torch.is_tensor(image):
                            # Make sure tensor is in CPU first
                            img_tensor = image.cpu()
                            # Check if we need to denormalize
                            if img_tensor.min() < 0:
                                img_tensor = denorm(img_tensor)
                            # Resize the tensor for wandb
                            img_tensor = image_resize(img_tensor)
                            # Convert to PIL
                            orig_small = transforms.ToPILImage()(img_tensor)
                        else:
                            # If already PIL
                            orig_small = image_resize(image)

                        # For generated image (already PIL format)
                        gen_small = image_resize(batch_images[0])

                        # Store for logging (using class index as key)
                        class_idx = label
                        wandb_originals[class_idx] = orig_small
                        wandb_generated[class_idx] = gen_small
                        wandb_class_names[class_idx] = class_name

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
                    class_name = self.datamodule.train_dataset.dataset.get_class_name(
                        labels[0].item()
                    )

                    # Save original and generated images
                    filename_generated = f"{class_name}_generated.png"
                    save_path_generated = f"{save_dir}/{filename_generated}"
                    batch_images[0].save(save_path_generated, "PNG")

                    filename_original = f"{class_name}_original.png"
                    save_path_original = f"{save_dir}/{filename_original}"
                    save_image(images[0].cpu(), save_path_original)

                    # For Wandb logging - prepare image pairs (original and generated)
                    if has_wandb:
                        # Convert tensor to PIL for wandb
                        if torch.is_tensor(images[0]):
                            # Make sure tensor is in CPU first
                            img_tensor = images[0].cpu()
                            # Check if we need to denormalize
                            if img_tensor.min() < 0:
                                img_tensor = denorm(img_tensor)
                            # Resize the tensor for wandb
                            img_tensor = image_resize(img_tensor)
                            # Convert to PIL
                            orig_small = transforms.ToPILImage()(img_tensor)
                        else:
                            # If already PIL
                            orig_small = image_resize(images[0])

                        # For generated image (PIL format)
                        gen_small = image_resize(batch_images[0])

                        # Store for logging (using class index as key)
                        class_idx = labels[0].item()
                        wandb_originals[class_idx] = orig_small
                        wandb_generated[class_idx] = gen_small
                        wandb_class_names[class_idx] = class_name

                # Save all generated images for this batch
                self.datamodule.train_dataset.dataset.image_storage.save_batch(
                    batch_images, cycle_idx, total_images_saved
                )
                total_images_saved += len(batch_images)

        print(f"Total images generated and saved: {total_images_saved}")

        # Log image examples to Wandb
        if (
            has_wandb
            and wandb_originals
            and hasattr(self.trainer_args.get("logger", None), "experiment")
        ):
            wandb_logger = self.trainer_args["logger"]

            # Log each class separately to avoid JSON serialization issues
            for class_idx in wandb_originals.keys():
                class_name = wandb_class_names[class_idx]

                wandb_logger.experiment.log(
                    {
                        f"cycle_{cycle_idx}/class_{class_name}/original": wandb.Image(
                            wandb_originals[class_idx]
                        ),
                        f"cycle_{cycle_idx}/class_{class_name}/generated": wandb.Image(
                            wandb_generated[class_idx]
                        ),
                        "cycle": cycle_idx,
                    }
                )

            # Log a summary of how many classes were augmented
            wandb_logger.experiment.log(
                {
                    f"cycle_{cycle_idx}/augmented_classes": len(wandb_originals),
                    "cycle": cycle_idx,
                }
            )
