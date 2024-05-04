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
import torch
from torch import nn
import torch.multiprocessing as mp

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


def init_datamodule(
    args: dict, checkpoint_filename: str, ssl_method: L.LightningModule, #TODO I think ssl_method should be removed from the args here?
) -> L.LightningDataModule:
    model_type = ModelTypes.get_model_type(args.model_name)
    ssl_method = SSLTypes.get_ssl_type(args.ssl_method)

    return ImbalancedImageNetDataModule(
        collate_fn=ssl_method.collate_fn,
        dataset_variant=ImageNetVariants.init_variant(args.imagenet_variant),
        imbalance_method=ImbalanceMethods.init_method(args.imbalance_method),
        splits=args.splits,
        batch_size=args.batch_size,
        resized_image_size=model_type.resized_image_size,
        checkpoint_filename=checkpoint_filename,
        transform=ssl_method.transforms,
    )


def init_model(args: dict, datamodule: L.LightningDataModule) -> nn.Module:
    model_type = ModelTypes.get_model_type(args.model_name)

    model_args = {
        "model_name": args.model_name,
        "resized_image_size": model_type.resized_image_size,
        "batch_size": args.batch_size,
        "output_size": datamodule.num_classes,
    }

    model = model_type.initialize(**model_args)

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


def init_ssl_type(args: dict, model: nn.Module, ipe) -> L.LightningModule:
    ssl_type = SSLTypes.get_ssl_type(args.ssl_method)
    ssl_args = {
        "model": model,
        "lr": args.lr,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_cycles * args.n_epochs_per_cycle,
        "parserargs": args,
        "ipe": ipe
    }

    return ssl_type.initialize(**ssl_args)


def run(args: dict, seed: int = 42) -> dict:
    set_seed(seed)

    checkpoint_filename = (
        args.model_name
        + "_"
        + args.imagenet_variant
        + "_"
        + args.imbalance_method
        + "-{epoch}-{val_loss:.2f}"
        if args.checkpoint is None
        else args.checkpoint
    )

    datamodule = init_datamodule(
        args,
        checkpoint_filename,
    )

    model = init_model(args, datamodule, len(datamodule.train_dataloader()))

    ssl_type = init_ssl_type(args, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        monitor="val_loss",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        mode="min",
    )

    tensorboard_logger = TensorBoardLogger("logs/", name=args.model_name)

    stats_monitor = DeviceStatsMonitor()

    callbacks = [checkpoint_callback, early_stopping_callback, stats_monitor]

    trainer_args = {
        "max_time": {"hours": args.max_hours_per_run},
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


def main():
    args = get_training_args()

    all_results = []

    for run_idx in range(args.num_runs):
        start_time = time.time()

        results = run(args, seed=run_idx)[0]

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({"training_time": training_time})

        print(results)

        all_results.append(results)

    print_mean_std(all_results)


if __name__ == "__main__":
    main()
