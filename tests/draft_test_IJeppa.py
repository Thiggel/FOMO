from experiment.models.SSLMethods.IJepa import IJepa

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiment.utils.get_training_args import get_training_args
from torch import nn
from datetime import date
import time
import argparse
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch.multiprocessing as mp

from experiment.utils.set_seed import set_seed
from experiment.utils.print_mean_std import print_mean_std
from experiment.utils.get_model_name import get_model_name

from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.models.ModelTypes import ModelTypes
from experiment.models.SSLTypes import SSLTypes
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.ImbalancedTraining import ImbalancedTraining

from experiment.models.SSLMethods.masks.multiblock import MaskCollator as MBMaskCollator
from experiment.dataset.transforms import make_transforms


@pytest.fixture
def model(monkeypatch, balanced_datamodule):
    # Define your model initialization here
    monkeypatch.setattr("sys.argv", ["program_name"])  # Provide some default value
    args = get_training_args()
    args.model_name = "ViTTinyJeppa"
    args.ssl_method = "I-Jepa"

    model = init_model(
        args,
    )
    ssl_type = init_ssl_type(args, model, len(balanced_datamodule.train_dataloader()))

    return ssl_type


@pytest.fixture
def balanced_datamodule(monkeypatch):
    monkeypatch.setattr("sys.argv", ["program_name"])
    args = get_training_args()
    args.model_name = "ViTTinyJeppa"
    args.ssl_method = "I-Jepa"
    collate_fn = MBMaskCollator(
        input_size=args.crop_size,
        patch_size=args.patch_size,
        pred_mask_scale=args.pred_mask_scale,
        enc_mask_scale=args.enc_mask_scale,
        aspect_ratio=args.aspect_ratio,
        nenc=args.num_enc_masks,
        npred=args.num_pred_masks,
        allow_overlap=args.allow_overlap,
        min_keep=args.min_keep,
    )

    transforms = make_transforms(
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        gaussian_blur=args.use_gaussian_blur,
        horizontal_flip=args.use_horizontal_flip,
        color_distortion=args.use_color_distortion,
        color_jitter=args.color_jitter_strength,
    )

    balanced_datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.NoImbalance,
        checkpoint_filename="test_balanced_dataset",
        collate_fn=collate_fn,
        transform=transforms,
    )
    return balanced_datamodule


def init_model(args: dict) -> nn.Module:
    if args.checkpoint is not None:
        model.load_state_dict(
            torch.load(
                args.checkpoint,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )["state_dict"]
        )

    return model


def init_ssl_type(
    args: dict, model: nn.Module, iterations_per_epoch
) -> L.LightningModule:
    ssl_type = SSLTypes.get_ssl_type(args.ssl_method)
    ssl_args = {
        "model": model,
        "lr": args.lr,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_cycles * args.n_epochs_per_cycle,
        "parserargs": args,
        "iterations_per_epoch": iterations_per_epoch,
    }

    return ssl_type.initialize(**ssl_args)


def model_training_step(model, balanced_datamodule):
    trainer = L.Trainer(max_steps=100)

    trainer.fit(model, balanced_datamodule)


def model_validation_step(model, balanced_datamodule):
    trainer = L.Trainer(max_steps=1)

    trainer.validate(model, balanced_datamodule)
    # Trainer.validate() should return 1 upon successful completio