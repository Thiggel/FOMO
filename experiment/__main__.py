from datetime import datetime
import time
import argparse
from argparse import Namespace
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn
import torch.multiprocessing as mp
from torchvision import transforms

from experiment.utils.set_seed import set_seed
from experiment.utils.print_mean_std import print_mean_std
from experiment.utils.get_training_args import get_training_args
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

mp.set_start_method("spawn")


def init_datamodule(args: dict, checkpoint_filename: str) -> L.LightningDataModule:
    ssl_method = SSLTypes.get_ssl_type(args.ssl_method)

    return ImbalancedImageNetDataModule(
        collate_fn=ssl_method.collate_fn(args),
        dataset_variant=ImageNetVariants.init_variant(args.imagenet_variant),
        imbalance_method=ImbalanceMethods.init_method(args.imbalance_method),
        splits=args.splits,
        batch_size=args.batch_size,
        checkpoint_filename=checkpoint_filename,
        transform=ssl_method.transforms(args),
        test_mode=args.test_mode,
    )


def init_model(args: Namespace, datamodule: L.LightningDataModule) -> nn.Module:
    model_type = ModelTypes.get_model_type(args.model_name)

    model_args = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "output_size": 128,  # simclear uses this hidden dim, vit doesnt use this parameter
        "image_size": args.crop_size,
        "classification_head": args.classification_head,
    }

    model = model_type.initialize(**model_args)

    return model


def init_ssl_type(
    args: Namespace, model: nn.Module, iterations_per_epoch
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


def run(args: Namespace, seed: int = 42) -> dict:
    set_seed(seed)

    checkpoint_filename = (
        args.model_name
        + "_"
        + args.imagenet_variant
        + "_"
        + args.imbalance_method
        + "_"
        + str(datetime.now())
    )

    datamodule = init_datamodule(
        args,
        checkpoint_filename,
    )

    model = init_model(args, datamodule)

    ssl_type = init_ssl_type(args, model, len(datamodule.train_dataloader()))

    if "loss" in args.early_stopping_monitor:
        mode = "min"
    if "acc" in args.early_stopping_monitor:
        mode = "max"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename + "-{epoch}-{val_loss:.2f}",
        monitor=args.early_stopping_monitor,
        mode=mode,
    )

    tensorboard_logger = TensorBoardLogger("logs/", name=args.model_name)

    stats_monitor = DeviceStatsMonitor()

    callbacks = [checkpoint_callback, stats_monitor]

    trainer_args = {
        "max_time": {"hours": int(args.max_hours_per_run) - 1},
        "max_epochs": args.n_epochs_per_cycle,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": tensorboard_logger if args.logger else None,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto",
    }

    imbalanced_training = ImbalancedTraining(
        args,
        trainer_args,
        ssl_type,
        datamodule,
        checkpoint_callback,
    )

    return imbalanced_training.run()


def set_checkpoint_for_run(args: Namespace, run_idx: int) -> str:
    if not hasattr(args, "checkpoint_list") or args.checkpoint_list is None:
        args.checkpoint_list = args.checkpoint

    if args.checkpoint_list is not None:
        args.checkpoint = args.checkpoint_list[run_idx % len(args.checkpoint_list)]

    return args


def run_different_seeds(args: Namespace) -> dict:
    all_results = []

    for run_idx in range(args.num_runs):
        start_time = time.time()

        run_args = set_checkpoint_for_run(args, run_idx)

        results = run(run_args, seed=args.seeds[run_idx])

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({"training_time": training_time})

        print(results)

        all_results.append(results)

    return all_results


def main():
    args = get_training_args()

    all_results = run_different_seeds(args)

    print_mean_std(all_results)


if __name__ == "__main__":
    main()
