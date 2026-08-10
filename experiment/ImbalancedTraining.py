import sys
import gc
from typing import Any, Optional
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import io
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset, random_split, Dataset, DataLoader
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
import re

from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.ood.ood import OOD
from diffusers import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionImageVariationPipeline,
    FluxPriorReduxPipeline,
    FluxPipeline,
)
from transformers import BlipForConditionalGeneration, BlipProcessor
from torchvision import transforms
import copy
import matplotlib.pyplot as plt

import os
import json
import pickle
import math
from itertools import zip_longest
from contextlib import contextmanager
import fcntl
import torch.distributed as dist

from experiment.dataset.ImageStorage import ImageStorage
from experiment.utils.get_num_workers import get_num_workers


class FluxAugmentor:
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        redux_model_id: str = "black-forest-labs/FLUX.1-Redux-dev",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            redux_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

    @torch.inference_mode()
    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        prompt: Optional[torch.Tensor] = None,
        num_steps: int = 6,
        guidance: float = 2.5,
    ):
        pipe_prior_output = self.pipe_prior_redux(image=images, prompt=prompt)

        prompt_embeds = pipe_prior_output.prompt_embeds
        pooled_prompt_embeds = pipe_prior_output.pooled_prompt_embeds

        if num_generations_per_image > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(
                num_generations_per_image, dim=0
            )
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(
                num_generations_per_image, dim=0
            )

        return self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=1,
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


class StableDiffusion3Augmentor:
    def __init__(self, device: Optional[str] = None, token: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token or os.getenv("HF_TOKEN")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            token=self.token,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        prompt: Optional[str] = "",
        num_steps: int = 20,
        guidance: float = 5.0,
        strength: float = 0.6,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        if isinstance(prompt, str):
            prompt = [prompt] * len(images)

        output = self.pipe(
            prompt=prompt,
            image=images,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_generations_per_image,
            height=height,
            width=width,
        )

        return output.images


class StableDiffusion3TextAugmentor:
    """VLM-captioned text-to-image control for image-conditioned repair.

    The selected image is used only to obtain an unlabeled caption. Generation
    then starts from noise, so this cleanly separates sparse-region allocation
    from SDEdit-style image conditioning.
    """

    def __init__(self, device: Optional[str] = None, token: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token or os.getenv("HF_TOKEN")
        self.caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            token=self.token,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        prompt: Optional[str] = None,
        num_steps: int = 20,
        guidance: float = 5.0,
        strength: float = 0.6,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        del strength
        if prompt is None:
            inputs = self.caption_processor(images=images, return_tensors="pt").to(
                self.device
            )
            captions = self.caption_model.generate(**inputs, max_new_tokens=32)
            prompts = self.caption_processor.batch_decode(
                captions, skip_special_tokens=True
            )
        elif isinstance(prompt, str):
            prompts = [prompt] * len(images)
        else:
            prompts = list(prompt)
        return self.pipe(
            prompt=prompts,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_generations_per_image,
            height=height,
            width=width,
        ).images


class StableDiffusion3CaptionedImg2ImgAugmentor:
    """BLIP-captioned SDEdit control paired with text-to-image generation."""

    def __init__(self, device: Optional[str] = None, token: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token or os.getenv("HF_TOKEN")
        self.caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            token=self.token,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        prompt: Optional[str] = None,
        num_steps: int = 20,
        guidance: float = 5.0,
        strength: float = 0.6,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        if prompt is None:
            inputs = self.caption_processor(images=images, return_tensors="pt").to(
                self.device
            )
            captions = self.caption_model.generate(**inputs, max_new_tokens=32)
            prompts = self.caption_processor.batch_decode(
                captions, skip_special_tokens=True
            )
        elif isinstance(prompt, str):
            prompts = [prompt] * len(images)
        else:
            prompts = list(prompt)
        return self.pipe(
            prompt=prompts,
            image=images,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_generations_per_image,
            height=height,
            width=width,
        ).images


class StrongConventionalAugmentor:
    """Offline RandAugment control with the same output cardinality as SD3."""

    def __init__(
        self,
        size: int,
        num_ops: int = 3,
        magnitude: int = 9,
    ):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.7, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=int(num_ops), magnitude=int(magnitude)
                ),
            ]
        )

    def augment(
        self,
        images,
        num_generations_per_image: int = 1,
        **kwargs,
    ):
        return [
            self.transform(image)
            for image in images
            for _ in range(num_generations_per_image)
        ]


class OODDistanceEpochLogger(L.Callback):
    def __init__(self, training_ref: "ImbalancedTraining", epoch_interval: int = 100):
        super().__init__()
        self.training_ref = training_ref
        self.epoch_interval = max(1, epoch_interval)

    def on_train_epoch_end(
        self, trainer, pl_module
    ) -> None:  # pragma: no cover - callback integration
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.epoch_interval != 0:
            return

        self.training_ref.log_average_ood_distance_for_epoch(current_epoch)


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
        self.total_epochs = args.total_epochs
        self.num_cycles = args.num_cycles
        self.n_epochs_per_cycle = args.n_epochs_per_cycle

        if self.num_cycles <= 0:
            raise ValueError("num_cycles must be a positive integer")

        if self.n_epochs_per_cycle is None:
            if self.total_epochs < self.num_cycles:
                raise ValueError("total_epochs must be >= num_cycles")
            if self.total_epochs % self.num_cycles != 0:
                raise ValueError("total_epochs must be divisible by num_cycles")
            self.n_epochs_per_cycle = self.total_epochs // self.num_cycles
        else:
            self.total_epochs = self.n_epochs_per_cycle * self.num_cycles

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
        self.initial_train_ds_size = (
            len(self.datamodule.train_dataset) if self.datamodule is not None else 0
        )
        self.added_indices = set()
        self.original_indices = set(range(self.initial_train_ds_size))
        self.last_ood_results: Optional[dict] = None
        self._frozen_selection_indices: Optional[list[int]] = None
        self.visualization_history = getattr(self.args, "visualization_history", 5)
        self.ood_distance_history: list[dict[str, Any]] = []
        self.avg_ood_distance_history: list[dict[str, Any]] = []
        self.max_ood_distance_history: list[dict[str, Any]] = []
        self.ood_distance_bins: Optional[np.ndarray] = None
        self.ood_class_metrics_history: list[dict[str, Any]] = []
        self.ood_class_sort_order: Optional[list[int]] = None
        self.class_distribution_history: list[dict[str, Any]] = []
        self.class_distribution_order: Optional[list[int]] = None
        self._offloaded_modules: list[tuple[str, torch.device]] = []
        self.enable_media_logging = bool(
            getattr(self.args, "enable_media_logging", False)
        )
        self.save_visualization_data = bool(
            getattr(self.args, "save_visualization_data", False)
        ) and self.enable_media_logging

        self._visualization_data_root = os.path.join(
            "visualizations",
            "data",
            self.checkpoint_filename,
            f"run_{self.run_idx + 1}",
        )
        if self.save_visualization_data:
            os.makedirs(self._visualization_data_root, exist_ok=True)

        self.num_workers = min(6, get_num_workers() // 2)
        self.class_distribution_workers = self._determine_class_distribution_workers()

        if self.datamodule is not None and self.args.logger:
            self._run_training_analysis(stage_label="start", cycle_reference=0)

    def _get_rank_info(self) -> tuple[int, int]:
        if (
            dist.is_available() and dist.is_initialized()
        ):  # pragma: no cover - distributed runtime
            return dist.get_rank(), dist.get_world_size()

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _is_primary_process(self) -> bool:
        rank, _ = self._get_rank_info()
        return rank == 0

    def _split_samples_for_rank(
        self, samples: list, world_size: int, rank: int
    ) -> tuple[list, int]:
        if world_size <= 1:
            return samples, 0

        total = len(samples)
        if total == 0:
            return [], 0

        per_rank = math.ceil(total / world_size)
        start = min(rank * per_rank, total)
        end = min(start + per_rank, total)
        return samples[start:end], start

    @contextmanager
    def _acquire_generation_lock(self, cycle_idx: int):
        lock_root = os.path.join(self.args.additional_data_path, "locks")
        os.makedirs(lock_root, exist_ok=True)
        lock_path = os.path.join(lock_root, f"cycle_{cycle_idx}.lock")

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _run_training_analysis(self, stage_label: str, cycle_reference: int) -> None:
        """Run configured analysis routines for a specific training stage."""

        if (
            not self.args.logger
            or not self.enable_media_logging
            or self.datamodule is None
        ):
            return

        if self.args.log_tsne:
            self.visualize_embedding_space(cycle_reference, stage_label=stage_label)

        if self.args.log_class_dist:
            self.save_class_dist(cycle_reference, stage_label=stage_label)

        self._log_generated_samples_summary(stage_label)

    def _get_module_device(
        self, module: Optional[torch.nn.Module]
    ) -> Optional[torch.device]:
        if module is None:
            return None

        module_device = getattr(module, "device", None)
        if isinstance(module_device, torch.device):
            return module_device
        if isinstance(module_device, str):
            return torch.device(module_device)

        for param in module.parameters():
            return param.device

        for buffer in module.buffers():
            return buffer.device

        return None

    def _offload_models_for_generation(self) -> None:
        if not torch.cuda.is_available():
            self._offloaded_modules = []
            return

        modules_to_offload: list[tuple[str, torch.nn.Module]] = []
        if hasattr(self, "ssl_method"):
            modules_to_offload.append(("ssl_method", self.ssl_method))

        self._offloaded_modules = []

        for attr_name, module in modules_to_offload:
            device = self._get_module_device(module)
            if device is None or device.type != "cuda":
                continue

            module.to("cpu")
            module_device_attr = getattr(type(module), "device", None)
            if module_device_attr is not None and hasattr(module, "device"):
                try:
                    setattr(module, "device", torch.device("cpu"))
                except AttributeError:
                    pass

            self._offloaded_modules.append((attr_name, device))

        torch.cuda.empty_cache()

    def _restore_offloaded_models(self) -> None:
        if not self._offloaded_modules:
            return

        for attr_name, device in self._offloaded_modules:
            module = getattr(self, attr_name, None)
            if module is None:
                continue

            target_device = device
            if target_device.type == "cuda" and not torch.cuda.is_available():
                target_device = torch.device("cpu")

            module.to(target_device)
            module_device_attr = getattr(type(module), "device", None)
            if module_device_attr is not None and hasattr(module, "device"):
                try:
                    setattr(module, "device", target_device)
                except AttributeError:
                    pass

        self._offloaded_modules = []

    def _log_generated_samples_summary(self, stage_label: str) -> None:
        """Log a summary of generated samples for the current stage."""

        if not (
            self.args.logger
            and self.enable_media_logging
            and self.args.log_generated_samples
        ):
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

    @staticmethod
    def _shade_color(
        color: str, alpha: float, lighten: float = 0.0
    ) -> tuple[float, ...]:
        import matplotlib.colors as mcolors

        base_rgb = np.array(mcolors.to_rgb(color))
        shaded_rgb = np.clip(
            base_rgb + (1.0 - base_rgb) * np.clip(lighten, 0.0, 1.0), 0.0, 1.0
        )
        return (*shaded_rgb, np.clip(alpha, 0.0, 1.0))

    def _save_visualization_data(self, relative_path: str, data: Any) -> str:
        if not self.save_visualization_data:
            return ""
        output_path = os.path.join(self._visualization_data_root, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(data, output_path)
        return output_path

    def _compute_sample_losses(
        self, dataset_indices: list[int]
    ) -> Optional[torch.Tensor]:
        if not dataset_indices:
            return None

        if not hasattr(self.ssl_method, "model"):
            print(
                "Warning: SSL method does not expose a model attribute for loss computation."
            )
            return None

        device = (
            self.ssl_method.device
            if hasattr(self.ssl_method, "device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = getattr(self.ssl_method, "dtype", torch.float32)

        # TADA's signal is learning difficulty, not a supervised class loss.
        # Estimate it here with the per-example symmetric InfoNCE loss between
        # two fresh SSL views.  The previous implementation incorrectly fed
        # projection features to cross-entropy with ground-truth labels.
        subset = Subset(self.datamodule.train_dataset, dataset_indices)
        base_dataset = subset.dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        old_transform = base_dataset.transform
        base_dataset.transform = self.datamodule.transform
        loader = DataLoader(
            subset,
            batch_size=min(len(dataset_indices), self.args.val_batch_size),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.datamodule.collate_fn,
        )

        was_training = self.ssl_method.training
        self.ssl_method.eval()
        model = self.ssl_method.model
        model_device = next(model.parameters(), torch.tensor(0)).device
        if model_device != device:
            model.to(device=device, dtype=dtype)

        losses: list[torch.Tensor] = []

        try:
            with torch.no_grad():
                for views, _ in loader:
                    if not isinstance(views, (tuple, list)) or len(views) < 2:
                        raise RuntimeError(
                            "early_loss requires an SSL transform with two views"
                        )
                    first = views[0].to(device=device, dtype=dtype)
                    second = views[1].to(device=device, dtype=dtype)
                    first_features = F.normalize(model(first), dim=-1)
                    second_features = F.normalize(model(second), dim=-1)
                    temperature = float(getattr(self.ssl_method, "temperature", lambda: 0.1)())
                    logits = first_features @ second_features.T / max(temperature, 1e-6)
                    targets = torch.arange(len(first_features), device=device)
                    batch_losses = 0.5 * (
                        F.cross_entropy(logits, targets, reduction="none")
                        + F.cross_entropy(logits.T, targets, reduction="none")
                    )
                    losses.append(batch_losses.detach().cpu())
        except Exception as exc:
            print(
                f"Warning: Failed to compute per-sample losses for OOD analysis: {exc}"
            )
            return None
        finally:
            base_dataset.transform = old_transform
            if was_training:
                self.ssl_method.train()

        if not losses:
            return None

        return torch.cat(losses)

    @staticmethod
    def _merge_bridge_tada_indices(
        bridge_indices: list[int],
        tada_ranked_indices: list[int],
        budget: int,
    ) -> list[int]:
        """Merge BRIDGE and TADA rankings under one exact repair budget."""
        if budget <= 0:
            return []

        bridge_quota = (budget + 1) // 2
        tada_quota = budget // 2
        selected: list[int] = []
        selected_set: set[int] = set()

        def append_unique(ranking: list[int], limit: int) -> None:
            added = 0
            for index in ranking:
                index = int(index)
                if index in selected_set:
                    continue
                selected.append(index)
                selected_set.add(index)
                added += 1
                if added >= limit or len(selected) >= budget:
                    return

        append_unique(bridge_indices, bridge_quota)
        append_unique(tada_ranked_indices, tada_quota)
        if len(selected) >= budget:
            return selected[:budget]

        # Overlap can leave one quota under-filled. Alternate the residual
        # rankings instead of silently reducing the requested repair volume.
        for pair in zip_longest(bridge_indices, tada_ranked_indices):
            for index in pair:
                if index is None or int(index) in selected_set:
                    continue
                selected.append(int(index))
                selected_set.add(int(index))
                if len(selected) >= budget:
                    return selected
        return selected

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
        return self._count_class_distribution_with_retry(
            dataset,
            desc_prefix="Counting class distribution",
            pin_memory=True,
        )

    def _determine_class_distribution_workers(self) -> int:
        """Choose the worker count for class distribution scans.

        Prefer the explicit ``class_distribution_workers`` config value when it
        is provided; otherwise fall back to the standard dataloader worker
        setting. Multiprocessing issues can still arise with some backends, so
        counting will retry with a single worker if necessary.
        """

        explicit_setting = getattr(self.args, "class_distribution_workers", None)
        if explicit_setting is not None:
            try:
                return max(0, int(explicit_setting))
            except (TypeError, ValueError):
                print(
                    "Invalid class_distribution_workers value; falling back to auto-detected setting."
                )

        return self.num_workers

    def _count_class_distribution_with_retry(
        self, dataset, desc_prefix: str, pin_memory: bool
    ) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else dataset.dataset.num_classes

        attempt_workers = self.class_distribution_workers
        fallback_used = False

        while True:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.val_batch_size,
                num_workers=attempt_workers,
                pin_memory=pin_memory,
            )

            class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

            try:
                for _, labels in tqdm(
                    dataloader,
                    desc=desc_prefix,
                ):
                    counts = torch.bincount(
                        labels.to(device), minlength=num_classes
                    )
                    class_counts += counts
                return class_counts
            except RuntimeError as exc:
                if fallback_used or attempt_workers == 0:
                    raise

                print(
                    f"{desc_prefix} failed with {attempt_workers} workers ({exc}). "
                    "Retrying with a single worker to avoid multiprocessing issues."
                )
                attempt_workers = 0
                fallback_used = True
            
            del dataloader
            gc.collect()

    def get_outliers(
        self, cycle_idx, precomputed_ood_indices: Optional[list[int]] = None
    ):
        if self.args.sample_selection == "ood":
            print("Using OOD detection for sample selection")
            if precomputed_ood_indices is not None:
                ood_indices = precomputed_ood_indices
            elif (
                getattr(self.args, "selection_reuse_policy", "adaptive")
                == "static_first_cycle"
                and self._frozen_selection_indices is not None
            ):
                ood_indices = self._frozen_selection_indices
            else:
                ood_indices = self.get_ood_indices(
                    self.datamodule.train_dataset, cycle_idx
                )
        elif self.args.sample_selection == "oracle":
            print("Using oracle indices for sample selection")
            ood_indices = self.get_oracle_indices()
        elif self.args.sample_selection == "oracle_real":
            # Real-data restoration oracle: restore previously withheld source
            # images, never duplicate the selected anchors.
            print("Using withheld real-source restoration oracle")
            ood_indices = self.get_oracle_real_indices()
        elif self.args.sample_selection == "early_loss":
            # TADA-style SSL learning-difficulty control.  It is a closest-
            # prior baseline rather than an unlabeled BRIDGE variant.
            print("Using highest SSL-loss examples for the TADA-style control")
            positions = list(range(len(self.datamodule.train_dataset)))
            losses = self._compute_sample_losses(positions)
            if losses is None:
                raise RuntimeError("Could not compute losses for early-loss selection")
            ood_indices = torch.argsort(losses, descending=True)[
                : self.args.num_ood_samples
            ].tolist()
        elif self.args.sample_selection == "bridge_tada":
            print(
                "Using a matched-budget hybrid of BRIDGE sparsity and "
                "TADA-style SSL difficulty"
            )
            bridge_indices = (
                list(precomputed_ood_indices)
                if precomputed_ood_indices is not None
                else self.get_ood_indices(
                    self.datamodule.train_dataset, cycle_idx
                )
            )
            positions = list(range(len(self.datamodule.train_dataset)))
            losses = self._compute_sample_losses(positions)
            if losses is None:
                raise RuntimeError(
                    "Could not compute losses for BRIDGE-TADA selection"
                )
            tada_ranked_indices = torch.argsort(
                losses, descending=True
            ).tolist()
            ood_indices = self._merge_bridge_tada_indices(
                bridge_indices,
                tada_ranked_indices,
                int(self.args.num_ood_samples),
            )
            if len(ood_indices) != int(self.args.num_ood_samples):
                raise RuntimeError(
                    "BRIDGE-TADA hybrid could not fill the requested "
                    f"budget: {len(ood_indices)}/"
                    f"{self.args.num_ood_samples}"
                )
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
            callbacks = list(cycle_trainer_args.get("callbacks", []))
            should_log_ood_epochs = False
            if self.args.logger:
                interval = getattr(self.args, "avg_ood_epoch_interval", 100)
                try:
                    interval = int(interval)
                except (TypeError, ValueError):
                    interval = 100

                if self.num_cycles <= 1 or getattr(self.args, "ssl", None) == "simclr":
                    should_log_ood_epochs = True

                if should_log_ood_epochs and not any(
                    isinstance(cb, OODDistanceEpochLogger) for cb in callbacks
                ):
                    callbacks.append(
                        OODDistanceEpochLogger(self, epoch_interval=max(1, interval))
                    )
            cycle_trainer_args["callbacks"] = callbacks

            # Each cycle owns exactly its configured update stage.  A new
            # Trainer is constructed for every cycle, so multiplying by
            # ``cycle_idx + 1`` previously produced E + 2E + ... training
            # epochs rather than C * E and invalidated compute matching.
            cycle_trainer_args["max_epochs"] = self.n_epochs_per_cycle
            max_steps_per_cycle = getattr(
                self.args, "max_steps_per_cycle", None
            )
            if max_steps_per_cycle is not None:
                cycle_trainer_args["max_steps"] = int(max_steps_per_cycle)
            if len(self.datamodule.val_dataset) == 0:
                cycle_trainer_args["limit_val_batches"] = 0
                cycle_trainer_args["num_sanity_val_steps"] = 0

            trainer = L.Trainer(**cycle_trainer_args)
            fit_kwargs = {
                "model": self.ssl_method,
                "datamodule": self.datamodule,
            }
            start_cycle = int(getattr(self.args, "start_cycle", 0) or 0)
            if (
                bool(getattr(self.args, "resume_trainer_state", False))
                and cycle_idx == start_cycle
                and self.args.checkpoint is not None
            ):
                fit_kwargs["ckpt_path"] = self.args.checkpoint
                fit_kwargs["weights_only"] = False

            skip_branch_stage = (
                cycle_idx == 0
                and bool(getattr(self.args, "skip_initial_training", False))
            )
            if skip_branch_stage:
                if self.args.checkpoint is None:
                    raise ValueError(
                        "skip_initial_training requires a frozen branch checkpoint"
                    )
                print(
                    "Skipping cycle-0 optimization: scoring the frozen common "
                    "branch checkpoint before the single repair stage."
                )
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.ssl_method.to(device)
            else:
                trainer.fit(**fit_kwargs)
                self._persist_latest_checkpoint(trainer, cycle_idx)

            if bool(
                getattr(self.args, "representation_diagnostics_each_cycle", False)
            ):
                self._save_cycle_representation_diagnostics(cycle_idx)

            wandb_logger = self.trainer_args.get("logger", None)
            has_wandb = (
                self.args.logger
                and wandb_logger
                and hasattr(wandb_logger, "experiment")
            )
            should_augment = (
                self.args.ood_augmentation
                and cycle_idx < self.num_cycles - 1
                and (
                    not bool(getattr(self.args, "repair_once", False))
                    or cycle_idx == 0
                )
            )
            run_ood_analysis = (has_wandb or should_augment) and (
                self.args.sample_selection != "oracle_real"
            )

            ssl_transform = None
            precomputed_ood_indices: Optional[list[int]] = None
            cycle_log_idx = cycle_idx + 1

            if run_ood_analysis:
                base_dataset = self.datamodule.train_dataset.dataset
                ssl_transform = copy.deepcopy(base_dataset.transform)
                base_dataset.transform = self.transform

                if (
                    getattr(self.args, "selection_reuse_policy", "adaptive")
                    == "static_first_cycle"
                    and self._frozen_selection_indices is not None
                ):
                    precomputed_ood_indices = self._frozen_selection_indices
                else:
                    precomputed_ood_indices = self.get_ood_indices(
                        self.datamodule.train_dataset, cycle_idx
                    )

                if has_wandb:
                    self._log_ood_distance_cdf(cycle_log_idx, wandb_logger)
                    self._log_ood_partition_summary(cycle_log_idx, wandb_logger)
                else:
                    self._log_average_ood_distance(
                        step_index=cycle_log_idx,
                        step_type="cycle",
                        label=f"Cycle {cycle_log_idx}",
                        wandb_logger=None,
                    )
                    self._log_max_ood_distance(
                        step_index=cycle_log_idx,
                        step_type="cycle",
                        label=f"Cycle {cycle_log_idx}",
                        wandb_logger=None,
                    )

            if not should_augment:
                if ssl_transform is not None:
                    self.datamodule.train_dataset.dataset.transform = ssl_transform
                if cycle_idx >= self.num_cycles - 1:
                    return
                print("OOD augmentation disabled, skipping generation")
                return

            if ssl_transform is None:
                ssl_transform = copy.deepcopy(
                    self.datamodule.train_dataset.dataset.transform
                )
                self.datamodule.train_dataset.dataset.transform = self.transform

            ood_indices = self.get_outliers(cycle_idx, precomputed_ood_indices)
            if (
                getattr(self.args, "selection_reuse_policy", "adaptive")
                == "static_first_cycle"
                and self._frozen_selection_indices is None
            ):
                self._frozen_selection_indices = list(ood_indices)

            anchor_dataset = (
                self.datamodule.train_dataset.dataset
                if self.args.sample_selection == "oracle_real"
                else self.datamodule.train_dataset
            )
            raw_ood_samples = [anchor_dataset[i] for i in ood_indices]
            # Teacher-cache runs append a stable underlying index as a third
            # item.  Generation and all existing operators intentionally use
            # the established (image, label) sample contract.
            ood_samples = [(sample[0], sample[1]) for sample in raw_ood_samples]
            ood_labels = torch.tensor(
                [label for _, label in tqdm(ood_samples, desc="Getting labels")]
            )

            print(f"Selected {len(ood_indices)} samples for augmentation")

            self._write_repair_manifest(
                cycle_idx=cycle_idx,
                selected_positions=ood_indices,
                labels=ood_labels.tolist(),
                anchor_samples=ood_samples,
                dataset=anchor_dataset,
                repair_operator=(
                    "oracle_real_restoration"
                    if self.args.sample_selection == "oracle_real"
                    else (
                        "anchor_duplicate"
                        if self.args.remove_diffusion
                        else str(self.args.generation_model)
                    )
                ),
            )

            if has_wandb:
                self._log_ood_class_distribution(
                    cycle_log_idx, ood_labels, wandb_logger, ood_indices
                )

            if self.args.remove_diffusion:
                # Size-matched anchor oversampling control: duplicate each selected
                # anchor as many times as the generative branch would add a variant.
                repeated_indices = (
                    ood_indices
                    if self.args.sample_selection == "oracle_real"
                    else [
                        index
                        for index in ood_indices
                        for _ in range(self.args.num_generations_per_ood_sample)
                    ]
                )
                self.added_indices.update(ood_indices)
                self.datamodule.add_samples_by_index(repeated_indices)
                print(
                    f"Added {len(repeated_indices)} anchor duplicates "
                    f"({len(ood_indices)} unique anchors) to the training set"
                )
            else:
                expected_new_images = (
                    len(ood_samples) * self.args.num_generations_per_ood_sample
                )
                if self._is_primary_process():
                    print(
                        f"\nGenerating {self.args.num_generations_per_ood_sample} images for each of {len(ood_samples)} OOD samples"
                    )
                    print(f"Expected total new images: {expected_new_images}")

                diffusion_pipe = None
                self._offload_models_for_generation()
                try:
                    if self.args.generation_model == "stable_diffusion":
                        diffusion_pipe = StableDiffusionAugmentor()
                    elif self.args.generation_model == "stable_diffusion_3":
                        diffusion_pipe = StableDiffusion3Augmentor()
                    elif self.args.generation_model == "stable_diffusion_3_t2i":
                        diffusion_pipe = StableDiffusion3TextAugmentor()
                    elif self.args.generation_model == "stable_diffusion_3_captioned_img2img":
                        diffusion_pipe = StableDiffusion3CaptionedImg2ImgAugmentor()
                    elif self.args.generation_model == "strong_augmentation":
                        diffusion_pipe = StrongConventionalAugmentor(
                            size=int(self.args.crop_size),
                            num_ops=int(self.args.strong_augmentation_num_ops),
                            magnitude=int(
                                self.args.strong_augmentation_magnitude
                            ),
                        )
                    elif self.args.generation_model == "flux":
                        diffusion_pipe = FluxAugmentor(
                            model_id=str(self.args.flux_model_id),
                            redux_model_id=str(self.args.flux_redux_model_id),
                        )
                    else:
                        raise ValueError(
                            f"Unknown generation model: {self.args.generation_model}"
                        )

                    self.generate_new_data(
                        ood_samples,
                        pipe=diffusion_pipe,
                        save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
                    )
                finally:
                    if torch.cuda.is_available():
                        if diffusion_pipe is not None:
                            del diffusion_pipe
                        torch.cuda.empty_cache()

                    self._restore_offloaded_models()

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

                if self.args.logger and self.args.log_generated_samples:
                    self._log_generated_samples_summary(
                        stage_label=f"cycle_{cycle_idx + 1}"
                    )

            # Reset transforms
            self.datamodule.train_dataset.dataset.transform = ssl_transform

        finally:
            self._cleanup_cycle_resources()
            gc.collect()
            torch.cuda.empty_cache()

    def run(self) -> dict:
        diagnostics = {}
        if self.args.pretrain:
            self.pretrain_imbalanced()

            if os.path.exists(self.checkpoint_callback.best_model_path):
                checkpoint = torch.load(
                    self.checkpoint_callback.best_model_path,
                    weights_only=False,
                )
                state_dict = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )
                self.ssl_method.load_state_dict(state_dict)

            if bool(getattr(self.args, "representation_diagnostics", True)):
                diagnostics = self.compute_representation_diagnostics()

        if self.datamodule is not None and self.args.logger:
            final_cycle_reference = (
                self.completed_cycles
                if self.completed_cycles
                else (self.num_cycles if self.num_cycles else 0)
            )
            self._run_training_analysis(
                stage_label="end", cycle_reference=final_cycle_reference
            )

        results = self.finetune() if self.args.finetune else {}
        results.update(diagnostics)
        return results

    def _persist_latest_checkpoint(self, trainer: L.Trainer, cycle_idx: int) -> None:
        """Overwrite ``last.ckpt`` with the model from the latest repair stage.

        BRIDGE constructs a fresh Lightning Trainer for every cycle. Reusing a
        ModelCheckpoint callback across those trainers can leave ``last.ckpt``
        pointing at an earlier stage. An explicit save after every completed
        fit makes post-hoc evaluation use the same final model that in-process
        evaluation sees.
        """
        checkpoint_dir = getattr(self.checkpoint_callback, "dirpath", None)
        if not checkpoint_dir:
            raise RuntimeError("Checkpoint callback does not define dirpath")
        checkpoint_path = os.path.join(str(checkpoint_dir), "last.ckpt")
        trainer.save_checkpoint(checkpoint_path)
        if self._is_primary_process():
            print(
                f"Saved completed cycle {cycle_idx + 1} checkpoint to "
                f"{checkpoint_path}"
            )

    def _write_repair_manifest(
        self,
        cycle_idx: int,
        selected_positions: list[int],
        labels: list[int],
        repair_operator: str,
        anchor_samples: Optional[list[tuple[torch.Tensor, Any]]] = None,
        dataset=None,
    ) -> None:
        """Record anchor ancestry for every generated/duplicated repair item."""
        dataset = dataset if dataset is not None else self.datamodule.train_dataset
        root = dataset
        while isinstance(root, Subset):
            root = root.dataset
        original_pool_size = len(getattr(root, "indices", []))
        variants = 1 if repair_operator == "oracle_real_restoration" else int(
            self.args.num_generations_per_ood_sample
        )
        output_dir = os.path.join(str(self.args.additional_data_path), "repair_manifests")
        anchor_dir = os.path.join(output_dir, f"cycle_{cycle_idx}_anchors")
        os.makedirs(anchor_dir, exist_ok=True)
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        rows = []
        for anchor_order, (position, label) in enumerate(
            zip(selected_positions, labels)
        ):
            node = dataset
            underlying = int(position)
            while isinstance(node, Subset):
                underlying = int(node.indices[underlying])
                node = node.dataset
            anchor_image_path = None
            if anchor_samples is not None and anchor_order < len(anchor_samples):
                image = anchor_samples[anchor_order][0]
                if torch.is_tensor(image):
                    image = ToPILImage()(inverse_normalize(image.detach().cpu()))
                anchor_image_path = os.path.join(anchor_dir, f"anchor_{anchor_order}.png")
                image.save(anchor_image_path, "PNG")
            for variant in range(variants):
                rows.append(
                    {
                        "repair_index": int(anchor_order * variants + variant),
                        "anchor_dataset_position": int(position),
                        "anchor_underlying_index": int(underlying),
                        "anchor_label": int(label),
                        "anchor_is_original": bool(
                            underlying < original_pool_size
                        ),
                        "anchor_image": anchor_image_path,
                        "variant": int(variant),
                    }
                )
        payload = {
            "cycle": int(cycle_idx),
            "repair_operator": repair_operator,
            "num_selected_anchors": int(len(selected_positions)),
            "num_repair_items": int(len(rows)),
            "generated_indexing": "repair_index matches HDF5 cycle-local index for generative operators",
            "rows": rows,
        }
        output_path = os.path.join(output_dir, f"cycle_{cycle_idx}.json")
        with open(output_path, "w") as handle:
            json.dump(payload, handle)
        print(f"Saved repair provenance manifest to {output_path}")

    # Do not use ``inference_mode`` here.  This diagnostic constructs a
    # multiprocessing DataLoader between training stages; workers forked while
    # inference mode is active can return inference tensors to the following
    # training stage, which then fail autograd with "Inference tensors cannot
    # be saved for backward".  ``no_grad`` provides the same memory benefit for
    # feature extraction without changing tensor provenance across stages.
    @torch.no_grad()
    def compute_representation_diagnostics(self) -> dict:
        """Scale-invariant health and geometry metrics on fixed original images."""
        import faiss

        train_dataset = self.datamodule.train_dataset
        if isinstance(train_dataset, Subset):
            base_dataset = train_dataset.dataset
            original_pool_size = len(getattr(base_dataset, "indices", []))
            original_positions = [
                position
                for position, dataset_index in enumerate(train_dataset.indices)
                if int(dataset_index) < original_pool_size
            ]
            panel_underlying_indices = np.asarray(
                [int(train_dataset.indices[position]) for position in original_positions],
                dtype=np.int64,
            )
            panel = Subset(train_dataset, original_positions)
        else:
            base_dataset = train_dataset
            panel = train_dataset
            panel_underlying_indices = np.arange(len(panel), dtype=np.int64)

        max_samples = int(
            getattr(self.args, "representation_diagnostics_max_samples", 20000)
        )
        if len(panel) > max_samples:
            generator = torch.Generator().manual_seed(0)
            chosen = torch.randperm(len(panel), generator=generator)[:max_samples]
            panel_underlying_indices = panel_underlying_indices[chosen.numpy()]
            panel = Subset(panel, chosen.tolist())

        old_transform = getattr(base_dataset, "transform", None)
        if hasattr(base_dataset, "transform"):
            base_dataset.transform = self.transform
        loader = DataLoader(
            panel,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=min(2, self.num_workers),
            pin_memory=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_method.model.to(device).eval()
        features, labels = [], []
        try:
            for loaded_batch in tqdm(loader, desc="Representation diagnostics"):
                images, targets = loaded_batch[0], loaded_batch[1]
                embeddings = self.ssl_method.model.extract_features(
                    images.to(device=device, dtype=self.ssl_method.dtype)
                )
                features.append(embeddings.float().cpu())
                labels.append(targets.cpu())
        finally:
            if hasattr(base_dataset, "transform"):
                base_dataset.transform = old_transform

        x = torch.cat(features)
        y = torch.cat(labels).numpy()
        feature_variance = x.var(dim=0, unbiased=False)
        normalized = F.normalize(x, dim=1).numpy().astype(np.float32)

        index = faiss.IndexFlatL2(normalized.shape[1])
        index.add(normalized)
        k = min(int(self.args.k) + 1, len(normalized))
        distances, neighbors = index.search(normalized, k)
        radii = distances[:, 1:].mean(axis=1)
        nearest_labels = y[neighbors[:, 1]]

        centered = torch.nan_to_num(x - x.mean(dim=0, keepdim=True))
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        try:
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        except torch.linalg.LinAlgError:
            # Rank-deficient collapsed representations are precisely what this
            # diagnostic must report.  Some CUDA eigensolvers reject their
            # covariance matrix, so derive the same non-zero spectrum from an
            # SVD rather than turning a completed experiment into a failure.
            singular_values = torch.linalg.svdvals(centered.float())
            eigenvalues = (
                singular_values.square() / max(1, len(centered) - 1)
            ).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        spectral_entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum()
        effective_rank = spectral_entropy.exp()

        sorted_radii = np.sort(radii)
        n = len(sorted_radii)
        gini = (
            (2 * np.arange(1, n + 1) - n - 1) @ sorted_radii
            / max(n * sorted_radii.sum(), 1e-12)
        )
        prefix = "representation/"
        metrics = {
            prefix + "feature_variance_mean": float(feature_variance.mean()),
            prefix + "feature_variance_min": float(feature_variance.min()),
            prefix + "effective_rank": float(effective_rank),
            prefix + "spectral_entropy": float(spectral_entropy),
            prefix + "knn_1_accuracy": float(np.mean(nearest_labels == y)),
            prefix + "normalized_radius_median": float(np.median(radii)),
            prefix + "normalized_radius_p90": float(np.quantile(radii, 0.90)),
            prefix + "normalized_radius_p95": float(np.quantile(radii, 0.95)),
            prefix + "radius_gini": float(gini),
            prefix + "num_original_samples": int(len(normalized)),
        }
        if bool(
            getattr(self.args, "representation_diagnostics_save_samples", False)
        ):
            self._last_representation_diagnostic_samples = {
                "underlying_indices": panel_underlying_indices,
                "labels": y.astype(np.int64, copy=False),
                "normalized_features": normalized.astype(np.float16, copy=False),
                "normalized_radii": radii.astype(np.float32, copy=False),
                "nearest_neighbor_indices": neighbors[:, 1].astype(
                    np.int64, copy=False
                ),
            }
        print("Representation diagnostics:", metrics)
        return metrics

    def _save_cycle_representation_diagnostics(self, cycle_idx: int) -> None:
        """Persist a fixed-original-panel geometry snapshot after each stage."""
        metrics = self.compute_representation_diagnostics()
        metrics.update(
            {
                "cycle": int(cycle_idx),
                "dataset_size_after_training": int(len(self.datamodule.train_dataset)),
                "stage": "after_training_before_next_repair",
            }
        )
        output_dir = os.path.join(
            str(self.args.additional_data_path), "representation_diagnostics"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cycle_{cycle_idx}.json")
        with open(output_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved fixed-original-panel diagnostics to {output_path}")
        samples = getattr(self, "_last_representation_diagnostic_samples", None)
        if samples is not None and bool(
            getattr(self.args, "representation_diagnostics_save_samples", False)
        ):
            sample_path = os.path.join(output_dir, f"cycle_{cycle_idx}_samples.npz")
            np.savez_compressed(sample_path, **samples)
            print(f"Saved fixed-panel sample diagnostics to {sample_path}")

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

    def get_oracle_real_indices(self) -> list:
        subset = self.datamodule.train_dataset
        if not isinstance(subset, Subset):
            raise ValueError("oracle_real requires an imbalanced Subset dataset")
        base = subset.dataset
        original = list(range(len(base.indices)))
        active = set(int(index) for index in subset.indices)
        withheld = [index for index in original if index not in active]
        requested = self.args.num_ood_samples * self.args.num_generations_per_ood_sample
        if len(withheld) < requested:
            raise ValueError(
                f"oracle_real needs {requested} withheld images, found {len(withheld)}"
            )
        generator = torch.Generator().manual_seed(int(self.args.seed or 0))
        chosen = torch.randperm(len(withheld), generator=generator)[:requested]
        return [withheld[index] for index in chosen.tolist()]

    def get_ood_indices(self, dataset, cycle_idx) -> list:
        """Get indices of OOD samples using feature-based detection"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_method.to(device)
        self.ssl_method.model.to(dtype=self.ssl_method.dtype)

        selection_dataset = dataset
        candidate_positions = None
        if bool(getattr(self.args, "selection_original_only", False)):
            if not isinstance(dataset, Subset):
                raise ValueError(
                    "selection_original_only requires the training dataset Subset"
                )
            base_dataset = dataset.dataset
            original_pool_size = len(getattr(base_dataset, "indices", []))
            candidate_positions = [
                position
                for position, dataset_index in enumerate(dataset.indices)
                if int(dataset_index) < original_pool_size
            ]
            selection_dataset = Subset(dataset, candidate_positions)

        feature_extractor = self.ssl_method.model.extract_features
        selection_dtype = self.ssl_method.dtype
        selection_model = None
        if str(getattr(self.args, "selection_encoder", "ssl")).lower() == "clip":
            from transformers import CLIPVisionModelWithProjection

            selection_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                torch_dtype=torch.float16,
            ).to(device).eval()
            image_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=device
            ).view(1, 3, 1, 1)
            image_std = torch.tensor(
                [0.229, 0.224, 0.225], device=device
            ).view(1, 3, 1, 1)
            clip_mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=device
            ).view(1, 3, 1, 1)
            clip_std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=device
            ).view(1, 3, 1, 1)

            def feature_extractor(batch):
                pixels = batch.float() * image_std + image_mean
                pixels = (pixels - clip_mean) / clip_std
                return selection_model(
                    pixel_values=pixels.to(dtype=torch.float16)
                ).image_embeds

            selection_dtype = torch.float32

        ood = OOD(
            args=self.args,
            dataset=selection_dataset,
            feature_extractor=feature_extractor,
            cycle_idx=cycle_idx,
            device=self.ssl_method.device,
            dtype=selection_dtype,
        )

        ood_indices = ood.ood()
        if candidate_positions is not None:
            ood_indices = [candidate_positions[index] for index in ood_indices]
            for key in (
                "selected_dataset_indices",
                "selected_dataset_indices_step",
                "top_dataset_indices",
            ):
                if ood.last_results is not None and key in ood.last_results:
                    ood.last_results[key] = [
                        candidate_positions[index]
                        for index in ood.last_results[key]
                    ]
        self.last_ood_results = ood.last_results
        if selection_model is not None:
            del selection_model
            gc.collect()
            torch.cuda.empty_cache()

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            original_pool_size = len(getattr(base_dataset, "indices", []))
            selected_underlying = [int(dataset.indices[index]) for index in ood_indices]
            original_count = sum(
                index < original_pool_size for index in selected_underlying
            )
            fraction = original_count / max(1, len(selected_underlying))
            print(
                f"Selected-anchor provenance: {original_count}/"
                f"{len(selected_underlying)} original ({fraction:.3f})"
            )
            if self.last_ood_results is not None:
                self.last_ood_results["selected_original_fraction"] = fraction

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
            for loaded_batch in tqdm(dataloader, desc="Collecting embeddings"):
                images, labels = loaded_batch[0], loaded_batch[1]
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
        color_map = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

        for cls in unique_classes:
            mask = (labels_np == cls) & (
                ~ood_mask_np if ood_mask_np is not None else True
            )
            plt.scatter(
                tsne_embeddings[mask, 0],
                tsne_embeddings[mask, 1],
                s=3,
                color=color_map[cls],
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
                color=color_map[cls],
                edgecolors="black",
                alpha=1.0,
            )

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        return plt.gcf()

    def visualize_embedding_space(
        self, cycle_idx: int, stage_label: Optional[str] = None
    ) -> None:
        if not self.enable_media_logging:
            return

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

        cycle_label = stage_label if stage_label is not None else cycle_idx
        cycle_label_str = str(cycle_label)
        self._save_visualization_data(
            os.path.join("tsne", f"cycle_{cycle_label_str}.pt"),
            {
                "cycle": cycle_label,
                "stage_label": stage_label,
                "tsne_embeddings": torch.tensor(tsne_embeddings, dtype=torch.float32),
                "labels": labels.cpu(),
                "ood_indices": ood_indices,
            },
        )

        fig = self.plot_tsne(tsne_embeddings, labels, class_names, ood_mask=ood_mask)

        vis_dir = f"{os.environ['BASE_CACHE_DIR']}/visualizations/tsne/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)
        png_path = f"{vis_dir}/tsne_cycle_{cycle_label}.png"
        pdf_path = f"{vis_dir}/tsne_cycle_{cycle_label}.pdf"
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
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
        if not self.enable_media_logging:
            return

        dataset = self.datamodule.train_dataset.dataset
        num_classes = dataset.num_classes
        class_counts_cpu = self._count_class_distribution_with_retry(
            self.datamodule.train_dataset,
            desc_prefix=(
                f"Counting class distribution for cycle {stage_label}"
                if stage_label is not None
                else f"Counting class distribution for cycle {cycle_idx}"
            ),
            pin_memory=False,
        ).cpu()
        self.class_counts = class_counts_cpu

        # Create dictionary mapping class indices to class names
        class_names_dict = {}
        for idx in range(num_classes):
            class_names_dict[idx] = (
                self.datamodule.train_dataset.dataset.get_class_name(idx)
            )

        # Save counts as tensor
        cycle_label = stage_label if stage_label is not None else cycle_idx
        self._save_visualization_data(
            os.path.join("class_distribution", f"counts_cycle_{cycle_label}.pt"),
            class_counts_cpu,
        )
        self._save_visualization_data(
            os.path.join("class_distribution", f"class_names_cycle_{cycle_label}.pt"),
            class_names_dict,
        )

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
            and self.enable_media_logging
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

                total_np = counts_np
                max_classes = getattr(self.args, "class_distribution_plot_topk", 30)
                sorted_indices = np.argsort(total_np)[::-1]

                if self.class_distribution_order is None:
                    self.class_distribution_order = (
                        sorted_indices[:max_classes].astype(int).tolist()
                    )
                else:
                    for idx in sorted_indices:
                        if (
                            int(idx) not in self.class_distribution_order
                            and len(self.class_distribution_order) < max_classes
                        ):
                            self.class_distribution_order.append(int(idx))
                    self.class_distribution_order = self.class_distribution_order[
                        :max_classes
                    ]

                order = self.class_distribution_order

                if self.class_distribution_history:
                    updated_history = []
                    for entry in self.class_distribution_history:
                        index_map = {
                            idx: pos for pos, idx in enumerate(entry["class_indices"])
                        }
                        total_vals = np.array(
                            [
                                float(entry["total"][index_map[idx]])
                                if idx in index_map
                                else 0.0
                                for idx in order
                            ],
                            dtype=float,
                        )
                        original_vals = np.array(
                            [
                                float(entry["original"][index_map[idx]])
                                if idx in index_map
                                else 0.0
                                for idx in order
                            ],
                            dtype=float,
                        )
                        generated_vals = np.array(
                            [
                                float(entry["generated"][index_map[idx]])
                                if idx in index_map
                                else 0.0
                                for idx in order
                            ],
                            dtype=float,
                        )
                        updated_history.append(
                            {
                                **entry,
                                "class_indices": list(order),
                                "class_names": [
                                    class_names_dict.get(idx, f"Class {idx}")
                                    for idx in order
                                ],
                                "total": total_vals,
                                "original": original_vals,
                                "generated": generated_vals,
                            }
                        )
                    self.class_distribution_history = updated_history

                order_array = np.array(order, dtype=int)
                class_names_ordered = [
                    class_names_dict.get(idx, f"Class {idx}") for idx in order
                ]
                total_ordered = total_np[order_array]
                original_ordered = original_np[order_array]
                generated_ordered = generated_np[order_array]

                new_entry = {
                    "cycle": cycle_label,
                    "class_indices": list(order),
                    "class_names": class_names_ordered,
                    "total": total_ordered.astype(float),
                    "original": original_ordered.astype(float),
                    "generated": generated_ordered.astype(float),
                }

                self.class_distribution_history.append(new_entry)
                if len(self.class_distribution_history) > self.visualization_history:
                    self.class_distribution_history = self.class_distribution_history[
                        -self.visualization_history :
                    ]

                self._save_visualization_data(
                    os.path.join("class_distribution", "history.pt"),
                    self.class_distribution_history,
                )

                # Create distribution plot with history overlay
                fig, ax = plt.subplots(figsize=(14, 7))
                x = np.arange(len(order), dtype=float)
                bar_width = 0.8
                plotted_cycles: set[Any] = set()

                history_to_plot = self.class_distribution_history[
                    -self.visualization_history :
                ]
                for idx, entry in enumerate(history_to_plot):
                    lighten = 0.5 * (
                        (len(history_to_plot) - idx - 1)
                        / max(len(history_to_plot) - 1, 1)
                        if len(history_to_plot) > 1
                        else 0.0
                    )
                    alpha = 0.85 if idx == len(history_to_plot) - 1 else 0.45
                    total_color = self._shade_color(
                        "tab:blue", alpha=alpha, lighten=lighten
                    )

                    label = None
                    if entry["cycle"] not in plotted_cycles:
                        label = f"Cycle {entry['cycle']}"
                        plotted_cycles.add(entry["cycle"])

                    # Ensure newer cycles are plotted behind older ones to show evolution
                    z_order = 1 + (len(history_to_plot) - 1 - idx)

                    ax.bar(
                        x,
                        entry["total"],
                        width=bar_width,
                        color=total_color,
                        label=label,
                        zorder=z_order,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(class_names_ordered, rotation=45, ha="right")
                ax.set_ylabel("Number of samples")
                ax.set_title(f"Class Distribution History - Cycle {cycle_label}")
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
                ax.legend(loc="upper right")
                fig.tight_layout()

                vis_dir = (
                    f"visualizations/class_distributions/{self.checkpoint_filename}"
                )
                os.makedirs(vis_dir, exist_ok=True)
                png_path = f"{vis_dir}/class_dist_cycle_{cycle_label}.png"
                pdf_path = f"{vis_dir}/class_dist_cycle_{cycle_label}.pdf"
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close(fig)

                wandb_logger = self.trainer_args["logger"]
                table = wandb.Table(
                    columns=[
                        "class",
                        "total_samples",
                        "original_samples",
                        "generated_samples",
                    ]
                )
                for name, total_val, original_val, generated_val in zip(
                    class_names_ordered,
                    total_ordered,
                    original_ordered,
                    generated_ordered,
                ):
                    table.add_data(
                        name,
                        float(total_val),
                        float(original_val),
                        float(generated_val),
                    )

                log_payload = {
                    f"class_distribution/cycle_{cycle_label}": wandb.Image(png_path),
                    f"class_distribution_table/cycle_{cycle_label}": table,
                    "cycle": cycle_label,
                    "analysis_stage": stage_label or "cycle",
                    "class_distribution_stats": {
                        "total_samples": counts_float.sum().item(),
                        "non_zero_classes": (class_counts_cpu > 0).sum().item(),
                        "mean_samples_per_class": counts_float.mean().item(),
                        "std_samples_per_class": counts_float.std().item(),
                    },
                }

                wandb_logger.experiment.log(log_payload)

                if hasattr(wandb, "Artifact") and hasattr(
                    wandb_logger.experiment, "log_artifact"
                ):
                    artifact_name = (
                        f"class_distribution_pdf_"
                        f"{self.checkpoint_filename}_cycle_{cycle_label}"
                    )
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="class_distribution_pdf",
                    )
                    artifact.add_file(pdf_path)
                    wandb_logger.experiment.log_artifact(
                        artifact,
                        aliases=[
                            f"cycle_{cycle_label}",
                            stage_label or "cycle",
                        ],
                    )
                else:
                    print(
                        "Warning: wandb version does not support Artifact logging;"
                        " skipping PDF upload."
                    )

            except Exception as e:
                print(f"Warning: Failed to log class distribution to wandb: {str(e)}")

    def pretrain_imbalanced(self) -> None:
        """Run the main training loop with OOD detection and augmentation"""
        if self.enable_media_logging:
            visualization_dir = (
                f"visualizations/class_distributions/{self.checkpoint_filename}"
            )
            os.makedirs(visualization_dir, exist_ok=True)

        start_cycle = int(getattr(self.args, "start_cycle", 0) or 0)
        stop_after_cycle = getattr(self.args, "stop_after_cycle", None)
        if stop_after_cycle is None:
            stop_after_cycle = self.num_cycles
        else:
            stop_after_cycle = min(self.num_cycles, int(stop_after_cycle))

        self.completed_cycles = start_cycle

        for cycle_idx in range(start_cycle, stop_after_cycle):
            print(f"Run {self.run_idx + 1}/{self.args.num_runs}")
            print(f"Pretraining cycle {cycle_idx + 1}/{self.num_cycles}")

            # Train for one cycle
            self.pretrain_cycle(cycle_idx)

            # Save and visualize class distribution
            if self.args.logger and self.enable_media_logging and self.args.log_tsne:
                self.visualize_embedding_space(cycle_idx + 1)
            if (
                self.args.logger
                and self.enable_media_logging
                and self.args.log_class_dist
            ):
                self.save_class_dist(cycle_idx + 1)

            self.completed_cycles = cycle_idx + 1

    def finetune(self) -> dict:
        """Run finetuning on benchmark datasets"""
        benchmarks = FinetuningBenchmarks.benchmarks
        requested_suite = getattr(self.args, "finetune_benchmark_suite", None)
        requested_benchmarks = getattr(self.args, "finetune_benchmarks", None)
        if requested_suite and requested_benchmarks:
            raise ValueError(
                "Set either finetune_benchmark_suite or finetune_benchmarks, "
                "not both"
            )
        if requested_suite:
            requested_benchmarks = (
                FinetuningBenchmarks.get_benchmark_suite_names(
                    str(requested_suite)
                )
            )
        if requested_benchmarks:
            requested_names = {str(name) for name in requested_benchmarks}
            known_names = {benchmark.__name__ for benchmark in benchmarks}
            unknown_names = sorted(requested_names - known_names)
            if unknown_names:
                raise ValueError(
                    "Unknown finetune benchmark(s): "
                    + ", ".join(unknown_names)
                    + ". Available benchmarks: "
                    + ", ".join(sorted(known_names))
                )
            benchmarks = [
                benchmark
                for benchmark in benchmarks
                if benchmark.__name__ in requested_names
            ]
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
                finetune_encoder=bool(
                    getattr(self.args, "finetune_encoder", False)
                ),
            )

            label_fraction = float(
                getattr(self.args, "finetune_label_fraction", 1.0)
            )
            if not 0.0 < label_fraction <= 1.0:
                raise ValueError(
                    "finetune_label_fraction must satisfy 0 < fraction <= 1"
                )
            if label_fraction < 1.0:
                finetuner.train_dataset = self._stratified_lowshot_subset(
                    finetuner.train_dataset,
                    label_fraction,
                    int(getattr(self.args, "finetune_seed", 0)),
                )
                print(
                    f"Using {len(finetuner.train_dataset)} examples "
                    f"({label_fraction:.1%}) for end-to-end low-shot evaluation"
                )

            configured_max_epochs = getattr(
                self.args, "finetune_max_epochs", None
            )
            self.trainer_args["max_epochs"] = (
                int(configured_max_epochs)
                if configured_max_epochs is not None
                else finetuner.max_epochs
            )

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
            max_time_minutes = getattr(
                self.args, "finetune_max_time_minutes", 25
            )
            self.trainer_args["max_time"] = {
                "minutes": int(max_time_minutes)
            }
            self.trainer_args["accumulate_grad_batches"] = 1

            if torch.cuda.is_available():
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

    @staticmethod
    def _dataset_labels(dataset) -> np.ndarray:
        if isinstance(dataset, Subset):
            parent = ImbalancedTraining._dataset_labels(dataset.dataset)
            return parent[np.asarray(dataset.indices, dtype=int)]
        # ImbalancedDataset exposes the original Hugging Face metadata through
        # attributes such as ``labels``.  For ImageNet those are class names,
        # not per-example integer targets, so resolve its explicit index/label
        # tensors before considering generic dataset attributes.
        if hasattr(dataset, "indices") and hasattr(dataset, "label_tensor"):
            labels = np.asarray(dataset.label_tensor, dtype=int)
            return labels[np.asarray(dataset.indices, dtype=int)]
        if hasattr(dataset, "samples"):
            return np.asarray([sample[1] for sample in dataset.samples], dtype=int)
        for attribute in ("targets", "labels", "label_tensor"):
            if hasattr(dataset, attribute):
                values = getattr(dataset, attribute)
                if torch.is_tensor(values):
                    values = values.cpu().numpy()
                values = np.asarray(values, dtype=int)
                if len(values) == len(dataset):
                    return values
        return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))])

    @classmethod
    def _stratified_lowshot_subset(cls, dataset, fraction, seed):
        labels = cls._dataset_labels(dataset)
        rng = np.random.default_rng(seed)
        selected = []
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            count = max(1, int(round(len(candidates) * fraction)))
            selected.extend(
                rng.choice(candidates, size=min(count, len(candidates)), replace=False)
                .astype(int)
                .tolist()
            )
        rng.shuffle(selected)
        return Subset(dataset, selected)

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
        num_classes = int(getattr(dataset, "num_classes", 0))
        if num_classes <= 0:
            return []

        if num_classes == 1:
            return [dataset.get_class_name(0)]

        label_tensor = torch.tensor(labels, dtype=torch.long)
        counts = torch.bincount(label_tensor, minlength=num_classes)
        nonzero_indices = torch.nonzero(counts, as_tuple=False).flatten()

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
        generated_counts = (total_counts.to(torch.long) - original_counts).clamp(min=0)

        return original_counts, generated_counts.to(device=total_counts.device)

    def _log_ood_class_distribution(
        self,
        cycle_idx: int,
        ood_labels: torch.Tensor,
        wandb_logger,
        ood_dataset_indices: list[int],
    ) -> None:
        if not self.enable_media_logging:
            return

        if ood_labels.numel() == 0 or not ood_dataset_indices:
            return

        if not self.last_ood_results:
            print(
                "Warning: Missing OOD statistics; skipping class distribution logging."
            )
            return

        dataset = self.datamodule.train_dataset.dataset
        distances = self.last_ood_results.get("distances")
        dataset_indices = self.last_ood_results.get("dataset_indices")

        if distances is None or dataset_indices is None:
            print(
                "Warning: OOD distances unavailable; skipping class distribution logging."
            )
            return

        index_to_distance = {
            int(idx): float(dist)
            for idx, dist in zip(dataset_indices, distances)
            if np.isfinite(dist)
        }

        filtered_records = [
            (
                idx,
                label.item() if torch.is_tensor(label) else int(label),
                index_to_distance.get(int(idx)),
            )
            for idx, label in zip(ood_dataset_indices, ood_labels)
        ]

        filtered_records = [
            (idx, label, dist)
            for idx, label, dist in filtered_records
            if dist is not None and np.isfinite(dist)
        ]

        if not filtered_records:
            print("Warning: No valid OOD distances found for selected samples.")
            return

        record_indices, record_labels, record_distances = zip(*filtered_records)
        labels_tensor = torch.tensor(record_labels, dtype=torch.long)
        distance_tensor = torch.tensor(record_distances, dtype=torch.float32)

        loss_tensor = self._compute_sample_losses([int(idx) for idx in record_indices])
        loss_available = loss_tensor is not None
        if loss_tensor is not None and len(loss_tensor) != len(distance_tensor):
            loss_tensor = None
            loss_available = False
            print(
                "Warning: Loss tensor size mismatch; ignoring loss statistics for OOD analysis."
            )

        unique_labels = torch.unique(labels_tensor)
        class_metrics: list[dict[str, Any]] = []

        for class_idx in unique_labels.tolist():
            mask = labels_tensor == class_idx
            if not mask.any():
                continue

            class_distances = distance_tensor[mask]
            avg_distance = float(class_distances.mean().item())
            avg_loss = (
                float(loss_tensor[mask].mean().item())
                if loss_tensor is not None
                else float("nan")
            )
            class_metrics.append(
                {
                    "class_idx": class_idx,
                    "class_name": dataset.get_class_name(class_idx),
                    "avg_distance": avg_distance,
                    "avg_loss": avg_loss,
                    "count": int(mask.sum().item()),
                }
            )

        if not class_metrics:
            return

        class_metrics.sort(key=lambda item: item["avg_distance"], reverse=True)
        top_k = min(20, len(class_metrics))
        selected_metrics = class_metrics[:top_k]

        order = [metric["class_idx"] for metric in selected_metrics]
        class_names = [metric["class_name"] for metric in selected_metrics]
        avg_distances = np.array(
            [metric["avg_distance"] for metric in selected_metrics], dtype=float
        )
        avg_losses = np.array(
            [metric["avg_loss"] for metric in selected_metrics], dtype=float
        )
        counts = np.array([metric["count"] for metric in selected_metrics], dtype=float)

        self.ood_class_sort_order = order
        self.ood_class_metrics_history = [
            {
                "cycle": cycle_idx,
                "class_indices": list(order),
                "class_names": class_names,
                "avg_distances": avg_distances,
                "avg_losses": avg_losses,
                "counts": counts,
                "loss_available": loss_available,
            }
        ]

        self._save_visualization_data(
            os.path.join("ood_class_metrics", f"cycle_{cycle_idx:04d}.pt"),
            {
                "cycle": cycle_idx,
                "class_indices": list(order),
                "class_names": class_names,
                "avg_distances": avg_distances.tolist(),
                "avg_losses": avg_losses.tolist(),
                "counts": counts.tolist(),
                "loss_available": loss_available,
            },
        )
        self._save_visualization_data(
            os.path.join("ood_class_metrics", "history.pt"),
            self.ood_class_metrics_history,
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = f"visualizations/ood_class_distributions/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)
        png_path = f"{vis_dir}/ood_class_dist_cycle_{cycle_idx}.png"
        pdf_path = f"{vis_dir}/ood_class_dist_cycle_{cycle_idx}.pdf"
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(order), dtype=float)
        bar_width = 0.4
        dist_color = self._shade_color("tab:blue", alpha=0.9, lighten=0.0)
        loss_color = self._shade_color("tab:orange", alpha=0.9, lighten=0.0)

        ax.bar(
            x - bar_width / 2,
            avg_distances,
            width=bar_width,
            color=dist_color,
            label="Avg distance",
            zorder=3,
        )

        if loss_available and not np.all(np.isnan(avg_losses)):
            ax.bar(
                x + bar_width / 2,
                avg_losses,
                width=bar_width,
                color=loss_color,
                label="Avg loss",
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_title(f"OOD Class Metrics - Avg Distance & Loss (Cycle {cycle_idx})")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        table = wandb.Table(columns=["class", "avg_distance", "avg_loss", "count"])
        for name, distance, loss, count in zip(
            class_names, avg_distances, avg_losses, counts
        ):
            table.add_data(
                name,
                float(distance),
                None if np.isnan(loss) else float(loss),
                int(count),
            )

        wandb_logger.experiment.log(
            {
                f"ood_class_distribution/cycle_{cycle_idx}": wandb.Image(png_path),
                f"ood_class_metrics_table/cycle_{cycle_idx}": table,
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
        table = (
            wandb.Table(columns=["OOD percentile", "Top classes"])
            if self.enable_media_logging
            else None
        )

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
            if table is not None:
                table.add_data(range_key, ", ".join(padded_top))

        self._save_visualization_data(
            os.path.join("ood_partition_summary", f"cycle_{cycle_idx:04d}.pt"),
            summary,
        )

        log_payload = {
            f"ood_partition_summary/cycle_{cycle_idx}": summary,
            "cycle": cycle_idx,
        }
        if table is not None:
            log_payload[f"ood_partition_summary_table/cycle_{cycle_idx}"] = table

        wandb_logger.experiment.log(log_payload)

    def _log_average_ood_distance(
        self,
        step_index: int,
        step_type: str,
        label: str,
        wandb_logger,
    ) -> None:
        if not self.last_ood_results:
            return

        distances = self.last_ood_results.get("distances")
        if distances is None or len(distances) == 0:
            return

        distances = np.asarray(distances, dtype=float)
        if distances.size == 0:
            return

        mean_distance = float(distances.mean())
        std_distance = float(distances.std())
        entry = {
            "step_index": int(step_index),
            "step_type": step_type,
            "label": label,
            "mean_distance": mean_distance,
            "std_distance": std_distance,
        }

        self.avg_ood_distance_history.append(entry)

        distances_tensor = torch.tensor(distances, dtype=torch.float32)
        filename_prefix = f"{step_type}_{step_index:04d}"
        self._save_visualization_data(
            os.path.join("ood_average_distance", f"{filename_prefix}_distances.pt"),
            distances_tensor,
        )
        self._save_visualization_data(
            os.path.join("ood_average_distance", "history.pt"),
            self.avg_ood_distance_history,
        )

        if wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_logger.experiment.log(
                {
                    "ood_average_distance/mean": mean_distance,
                    "ood_average_distance/std": std_distance,
                    "ood_average_distance/step_label": label,
                }
            )

        if not self.enable_media_logging:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_positions = np.arange(1, len(self.avg_ood_distance_history) + 1)
        means = [entry["mean_distance"] for entry in self.avg_ood_distance_history]
        labels = [entry["label"] for entry in self.avg_ood_distance_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_positions, means, marker="o", linewidth=2, color="tab:purple")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Average OOD distance")
        ax.set_title("Average OOD Distance Over Time")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        vis_dir = os.path.join(
            "visualizations",
            "ood_average_distance",
            self.checkpoint_filename,
        )
        os.makedirs(vis_dir, exist_ok=True)
        image_path = os.path.join(vis_dir, "avg_ood_distance.png")
        pdf_path = os.path.join(vis_dir, "avg_ood_distance.pdf")
        fig.savefig(image_path, dpi=120, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        if wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_logger.experiment.log(
                {
                    "ood_average_distance_plot": wandb.Image(fig),
                    "ood_average_distance/step_label": label,
                }
            )

        plt.close(fig)

        if (
            wandb_logger
            and hasattr(wandb_logger, "experiment")
            and hasattr(wandb_logger.experiment, "log_artifact")
        ):
            artifact_name = self._sanitize_artifact_name(
                f"{self.checkpoint_filename}_ood_average_distance"
            )
            artifact = wandb.Artifact(name=artifact_name, type="visualization")
            artifact.add_file(pdf_path, name=os.path.basename(pdf_path))
            wandb_logger.experiment.log_artifact(artifact)

    def _log_max_ood_distance(
        self,
        step_index: int,
        step_type: str,
        label: str,
        wandb_logger,
    ) -> None:
        if not self.last_ood_results:
            return

        distances = self.last_ood_results.get("distances")
        if distances is None or len(distances) == 0:
            return

        distances = np.asarray(distances, dtype=float)
        if distances.size == 0:
            return

        max_distance = float(distances.max())
        entry = {
            "step_index": int(step_index),
            "step_type": step_type,
            "label": label,
            "max_distance": max_distance,
        }

        self.max_ood_distance_history.append(entry)

        distances_tensor = torch.tensor(distances, dtype=torch.float32)
        filename_prefix = f"{step_type}_{step_index:04d}"
        self._save_visualization_data(
            os.path.join("ood_max_distance", f"{filename_prefix}_distances.pt"),
            distances_tensor,
        )
        self._save_visualization_data(
            os.path.join("ood_max_distance", "history.pt"),
            self.max_ood_distance_history,
        )

        if wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_logger.experiment.log(
                {
                    "ood_max_distance/max": max_distance,
                    "ood_max_distance/step_label": label,
                }
            )

        if not self.enable_media_logging:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_positions = np.arange(1, len(self.max_ood_distance_history) + 1)
        maxima = [entry["max_distance"] for entry in self.max_ood_distance_history]
        labels = [entry["label"] for entry in self.max_ood_distance_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_positions, maxima, marker="o", linewidth=2, color="tab:red")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Maximum OOD distance")
        ax.set_title("Maximum OOD Distance Over Time")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        vis_dir = os.path.join(
            "visualizations",
            "ood_max_distance",
            self.checkpoint_filename,
        )
        os.makedirs(vis_dir, exist_ok=True)
        image_path = os.path.join(vis_dir, "max_ood_distance.png")
        pdf_path = os.path.join(vis_dir, "max_ood_distance.pdf")
        fig.savefig(image_path, dpi=120, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        if wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_logger.experiment.log(
                {
                    "ood_max_distance_plot": wandb.Image(fig),
                    "ood_max_distance/step_label": label,
                }
            )

        plt.close(fig)

        if (
            wandb_logger
            and hasattr(wandb_logger, "experiment")
            and hasattr(wandb_logger.experiment, "log_artifact")
        ):
            artifact_name = self._sanitize_artifact_name(
                f"{self.checkpoint_filename}_ood_max_distance"
            )
            artifact = wandb.Artifact(name=artifact_name, type="visualization")
            artifact.add_file(pdf_path, name=os.path.basename(pdf_path))
            wandb_logger.experiment.log_artifact(artifact)

    def log_average_ood_distance_for_epoch(self, epoch: int) -> None:
        wandb_logger = self.trainer_args.get("logger", None)
        if not (
            self.args.logger
            and wandb_logger
            and hasattr(wandb_logger, "experiment")
            and self.datamodule is not None
        ):
            return

        train_dataset = getattr(self.datamodule, "train_dataset", None)
        if train_dataset is None:
            return

        base_dataset = train_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset

        old_transform = base_dataset.transform
        base_dataset.transform = self.transform

        try:
            self.get_ood_indices(train_dataset, f"epoch_{epoch}")
            self._log_average_ood_distance(
                step_index=epoch,
                step_type="epoch",
                label=f"Epoch {epoch}",
                wandb_logger=wandb_logger,
            )
            self._log_max_ood_distance(
                step_index=epoch,
                step_type="epoch",
                label=f"Epoch {epoch}",
                wandb_logger=wandb_logger,
            )
        finally:
            base_dataset.transform = old_transform

    def _log_ood_distance_cdf(self, cycle_idx: int, wandb_logger) -> None:
        if not self.last_ood_results:
            return

        distances = self.last_ood_results.get("distances")

        if distances is None or len(distances) == 0:
            return

        distances = np.asarray(distances, dtype=float)
        if distances.size == 0:
            return

        history_entry = {"cycle": cycle_idx, "distances": distances}
        self.ood_distance_history.append(history_entry)
        if len(self.ood_distance_history) > self.visualization_history:
            self.ood_distance_history = self.ood_distance_history[
                -self.visualization_history :
            ]

        self._save_visualization_data(
            os.path.join(
                "ood_distance_distribution",
                f"cycle_{cycle_idx:04d}_distances.pt",
            ),
            torch.tensor(distances, dtype=torch.float32),
        )
        self._save_visualization_data(
            os.path.join("ood_distance_distribution", "history.pt"),
            self.ood_distance_history,
        )

        all_distances = np.concatenate(
            [entry["distances"] for entry in self.ood_distance_history]
        )

        if all_distances.size == 0:
            return

        min_val = float(all_distances.min())
        max_val = float(all_distances.max())
        if np.isclose(min_val, max_val):
            bins = np.linspace(min_val - 1e-6, max_val + 1e-6, num=10)
        else:
            bins = np.histogram_bin_edges(all_distances, bins="auto")
            if bins.size < 5:
                bins = np.linspace(min_val, max_val, num=10)
        self.ood_distance_bins = bins

        # Determine a reasonable right-side cutoff so the plot focuses on the modes.
        cutoff_density_threshold = 0.005
        histogram_density, bin_edges = np.histogram(
            all_distances, bins=self.ood_distance_bins, density=True
        )
        above_threshold_indices = np.where(
            histogram_density >= cutoff_density_threshold
        )[0]
        right_cutoff = None
        if above_threshold_indices.size > 0:
            last_idx = int(above_threshold_indices[-1])
            # Extend the limit to the edge of the last bin that meets the threshold
            right_cutoff = float(bin_edges[min(last_idx + 1, len(bin_edges) - 1)])

        stats_payload = {
            "mean": float(distances.mean()),
            "std": float(distances.std()),
            "min": float(distances.min()),
            "max": float(distances.max()),
        }

        if not self.enable_media_logging:
            wandb_logger.experiment.log(
                {
                    f"ood_distance_stats/cycle_{cycle_idx}": stats_payload,
                    "cycle": cycle_idx,
                }
            )
            self._log_average_ood_distance(
                step_index=cycle_idx,
                step_type="cycle",
                label=f"Cycle {cycle_idx}",
                wandb_logger=wandb_logger,
            )
            self._log_max_ood_distance(
                step_index=cycle_idx,
                step_type="cycle",
                label=f"Cycle {cycle_idx}",
                wandb_logger=wandb_logger,
            )
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = f"visualizations/ood_distance_distribution/{self.checkpoint_filename}"
        os.makedirs(vis_dir, exist_ok=True)

        pdf_path = f"{vis_dir}/ood_distance_distribution_cycle_{cycle_idx}.pdf"

        fig, ax = plt.subplots(figsize=(12, 6))
        history_to_plot = self.ood_distance_history[-self.visualization_history :]

        for idx, entry in enumerate(history_to_plot):
            lighten = 0.5 * (
                (len(history_to_plot) - idx - 1) / max(len(history_to_plot) - 1, 1)
                if len(history_to_plot) > 1
                else 0.0
            )
            alpha = 0.7 if idx == len(history_to_plot) - 1 else 0.3
            color = self._shade_color("tab:purple", alpha=alpha, lighten=lighten)

            ax.hist(
                entry["distances"],
                bins=self.ood_distance_bins,
                density=True,
                color=color,
                label=f"Cycle {entry['cycle']}",
                histtype="stepfilled",
                edgecolor=self._shade_color(
                    "tab:purple", alpha=min(alpha + 0.2, 1.0), lighten=lighten / 2
                ),
                linewidth=1.0,
            )

        ax.set_xlabel("OOD distance")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of OOD Distances - Cycle {cycle_idx}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper right")
        # Ensure the distribution doesn't show negative distances, which are not meaningful.
        left_limit = 0.0
        current_left, current_right = ax.get_xlim()
        # Preserve any existing right-side adjustments while enforcing the left boundary at 0.
        ax.set_xlim(left=left_limit, right=current_right)

        if right_cutoff is not None:
            # Add a small padding so the final bar is fully visible.
            padding = max((right_cutoff - left_limit) * 0.01, 1e-6)
            ax.set_xlim(left=left_limit, right=right_cutoff + padding)
        fig.tight_layout()

        hist_image = wandb.Image(fig)
        wandb_logger.experiment.log(
            {
                f"ood_distance_distribution/cycle_{cycle_idx}": hist_image,
                "cycle": cycle_idx,
                f"ood_distance_stats/cycle_{cycle_idx}": stats_payload,
            }
        )

        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        if hasattr(wandb_logger.experiment, "log_artifact"):
            artifact_name = self._sanitize_artifact_name(
                f"{self.checkpoint_filename}_cycle_{cycle_idx}_ood_distance_distribution"
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="visualization",
                metadata={"cycle": cycle_idx},
            )
            artifact.add_file(pdf_path, name=os.path.basename(pdf_path))
            wandb_logger.experiment.log_artifact(artifact)

        self._log_average_ood_distance(
            step_index=cycle_idx,
            step_type="cycle",
            label=f"Cycle {cycle_idx}",
            wandb_logger=wandb_logger,
        )
        self._log_max_ood_distance(
            step_index=cycle_idx,
            step_type="cycle",
            label=f"Cycle {cycle_idx}",
            wandb_logger=wandb_logger,
        )

    def generate_new_data(self, ood_samples, pipe, save_subfolder) -> None:
        """
        Generate new data using the configured diffusion model and log examples to Wandb.
        Stable Diffusion 3 supports batching, while Flux runs one image at a time for
        reliability.
        """
        cycle_idx = int(save_subfolder.split("/")[-1])
        rank, world_size = self._get_rank_info()
        is_primary_rank = rank == 0
        total_samples = len(ood_samples)
        local_samples, shard_start = self._split_samples_for_rank(
            ood_samples, world_size, rank
        )

        if not local_samples:
            if is_primary_rank:
                print("No OOD samples assigned for generation.")
            return

        generations_per_sample = self.args.num_generations_per_ood_sample
        global_total_expected = total_samples * generations_per_sample

        denorm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )

        # For Wandb logging
        has_wandb = (
            is_primary_rank
            and self.args.logger
            and self.enable_media_logging
            and self.args.log_generated_samples
        )
        image_resize = transforms.Resize((64, 64))  # Resize to 64x64 for Wandb
        wandb_logger = self.trainer_args.get("logger", None)
        log_every_n_classes = getattr(self.args, "log_every_n_classes", 1)
        max_wandb_pairs = getattr(self.args, "max_wandb_augmentation_pairs", 8)
        target_wandb_pairs = max(4, max_wandb_pairs)
        logged_classes = 0
        wandb_pairs = []

        total_augmented_classes = len(
            {
                int(label.item() if torch.is_tensor(label) else label)
                for _, label in ood_samples
            }
        )
        next_storage_index = shard_start * generations_per_sample
        local_images_saved = 0
        already_saved_sample_classes = set()
        dataset_obj = self.datamodule.train_dataset.dataset

        def _save_images_with_lock(
            images_to_save: list[Image.Image], offset: int
        ) -> None:
            if not images_to_save:
                return
            with self._acquire_generation_lock(cycle_idx):
                dataset_obj.image_storage.save_batch(images_to_save, cycle_idx, offset)

        def _log_sample_outputs(
            original_image, generated_images, label, class_name: str
        ) -> None:
            nonlocal logged_classes
            orig_small = None
            gen_small = None
            if has_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
                orig_small, gen_small = self._prepare_wandb_image_pair(
                    original_image, generated_images[0], denorm, image_resize
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
                and label not in already_saved_sample_classes
            ):
                already_saved_sample_classes.add(label)

                save_dir = (
                    f"{os.environ['BASE_CACHE_DIR']}/ood_samples/cycle_{cycle_idx}/"
                )
                os.makedirs(save_dir, exist_ok=True)

                filename_generated = f"{class_name}_generated.png"
                save_path_generated = f"{save_dir}/{filename_generated}"
                generated_images[0].save(save_path_generated, "PNG")

                filename_original = f"{class_name}_original.png"
                save_path_original = f"{save_dir}/{filename_original}"

                if torch.is_tensor(original_image):
                    save_image(original_image.cpu(), save_path_original)
                else:
                    original_image.save(save_path_original, "PNG")

        if self.args.generation_model == "flux":
            flux_batch_size = max(1, getattr(self.args, "flux_batch_size", 1))
            flux_guidance = getattr(self.args, "flux_guidance", 2.5)
            flux_num_steps = getattr(self.args, "flux_num_steps", 6)
            dataloader = DataLoader(
                local_samples,
                batch_size=flux_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )

            processed = 0
            for images, labels in tqdm(
                dataloader,
                desc=f"Generating New Data with Flux... (rank {rank})",
                disable=not is_primary_rank,
            ):
                batch = [ToPILImage()(denorm(image)) for image in images]

                expected = len(batch) * generations_per_sample
                generated_images = []
                max_flux_attempts = 3
                for attempt in range(max_flux_attempts):
                    if len(generated_images) >= expected:
                        break
                    chunk = pipe.augment(
                        batch,
                        num_generations_per_image=generations_per_sample,
                        num_steps=flux_num_steps,
                        guidance=flux_guidance,
                    )
                    generated_images.extend(chunk)
                if len(generated_images) < expected:
                    raise RuntimeError(
                        "Flux generation returned an unexpected number of images "
                        f"(expected {expected}, got {len(generated_images)}) even after {max_flux_attempts} attempts."
                    )
                generated_images = generated_images[:expected]


                for sample_idx, (image, label) in enumerate(zip(images, labels)):
                    label_int = int(label.item() if torch.is_tensor(label) else label)
                    class_name = dataset_obj.get_class_name(label_int)
                    start = sample_idx * generations_per_sample
                    end = start + generations_per_sample
                    sample_generated_images = generated_images[start:end]

                    _log_sample_outputs(
                        image,
                        sample_generated_images,
                        label_int,
                        class_name,
                    )

                    _save_images_with_lock(
                        sample_generated_images,
                        next_storage_index,
                    )
                    next_storage_index += len(sample_generated_images)
                    local_images_saved += len(sample_generated_images)

                processed += len(batch)
                if processed % 10 == 0 and is_primary_rank:
                    print(
                        f"Processed {processed}/{len(local_samples)} images on rank {rank}, generated {local_images_saved} augmentations"
                    )
        elif self.args.generation_model in {
            "stable_diffusion_3",
            "stable_diffusion_3_t2i",
            "stable_diffusion_3_captioned_img2img",
            "strong_augmentation",
        }:
            sd3_batch_size = max(1, getattr(self.args, "sd3_batch_size", 1))
            sd3_guidance = getattr(self.args, "sd3_guidance", 5.0)
            sd3_num_steps = getattr(self.args, "sd3_num_steps", 20)
            sd3_strength = getattr(self.args, "sd3_strength", 0.6)
            sd3_resolution = int(self.args.crop_size)

            dataloader = DataLoader(
                local_samples,
                batch_size=sd3_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )

            for batch_idx, (images, labels) in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Generating New Data with Stable Diffusion 3... (rank {rank})",
                    disable=not is_primary_rank,
                )
            ):
                batch = [ToPILImage()(denorm(image)) for image in images]

                generated_images = pipe.augment(
                    batch,
                    num_generations_per_image=generations_per_sample,
                    num_steps=sd3_num_steps,
                    guidance=sd3_guidance,
                    strength=sd3_strength,
                    height=sd3_resolution,
                    width=sd3_resolution,
                )

                expected = len(batch) * generations_per_sample
                if len(generated_images) != expected:
                    raise RuntimeError(
                        "Stable Diffusion 3 generation returned an unexpected number of images "
                        f"(expected {expected}, got {len(generated_images)})."
                    )

                per_sample_generated = []
                for sample_idx in range(len(batch)):
                    start = sample_idx * self.args.num_generations_per_ood_sample
                    end = start + self.args.num_generations_per_ood_sample
                    per_sample_generated.append(generated_images[start:end])

                for image, label, generated in zip(
                    images, labels, per_sample_generated
                ):
                    label_int = int(label.item() if torch.is_tensor(label) else label)
                    class_name = dataset_obj.get_class_name(label_int)
                    _log_sample_outputs(image, generated, label_int, class_name)

                _save_images_with_lock(generated_images, next_storage_index)
                next_storage_index += len(generated_images)
                local_images_saved += len(generated_images)

                if batch_idx % 10 == 0 and is_primary_rank:
                    processed = min(
                        (batch_idx + 1) * sd3_batch_size, len(local_samples)
                    )
                    print(
                        f"Processed {processed}/{len(local_samples)} images on rank {rank}, generated {local_images_saved} augmentations"
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
                local_samples,
                batch_size=effective_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )

            for batch_idx, (images, labels) in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Generating New Data with Stable Diffusion... (rank {rank})",
                    disable=not is_primary_rank,
                )
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
                if has_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
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
                _save_images_with_lock(batch_images, next_storage_index)
                next_storage_index += len(batch_images)
                local_images_saved += len(batch_images)

        if is_primary_rank:
            print(f"Total images generated and saved: {global_total_expected}")

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
                    f"cycle_{cycle_idx}/augmented_classes": total_augmented_classes,
                    "cycle": cycle_idx,
                }
            )
