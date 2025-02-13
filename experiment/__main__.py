from datetime import datetime
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
import os

from dotenv import load_dotenv
import time
import argparse
from argparse import Namespace
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
from torch import nn
import torch.multiprocessing as mp
from torchvision import transforms
from huggingface_hub import login

from experiment.utils.set_seed import set_seed
from experiment.utils.print_mean_std import print_mean_std
from experiment.utils.get_training_args import get_training_args
from experiment.utils.get_model_name import get_model_name
from experiment.utils.generate_random_string import generate_random_string
from experiment.utils.calc_novelty_score import calc_novelty_score

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
torch.multiprocessing.set_sharing_strategy("file_system")
import os
import shutil
import torch.multiprocessing as mp
from pathlib import Path


def init_datamodule(args: dict, checkpoint_filename: str) -> L.LightningDataModule:
    ssl_method = SSLTypes.get_ssl_type(args.ssl_method)

    return ImbalancedImageNetDataModule(
        collate_fn=ssl_method.collate_fn(args),
        dataset_variant=ImageNetVariants.init_variant(args.imagenet_variant),
        imbalance_method=ImbalanceMethods.init_method(args.imbalance_method),
        splits=args.splits,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        checkpoint_filename=checkpoint_filename,
        transform=ssl_method.transforms(args),
        test_mode=args.test_mode,
        additional_data_path=args.additional_data_path,
    )


def init_model(args: Namespace) -> nn.Module:
    model_type = ModelTypes.get_model_type(args.model_name)

    model_args = {
        "model_name": args.model_name,
        "batch_size": args.train_batch_size,
        "output_size": 128,  # simclear uses this hidden dim, vit doesnt use this parameter
        "image_size": args.crop_size,
        "classification_head": args.classification_head,
    }

    model = model_type.initialize(**model_args)

    return model


def init_ssl_type(
    args: Namespace,
    model: nn.Module,
) -> L.LightningModule:
    ssl_type = SSLTypes.get_ssl_type(args.ssl_method)
    ssl_args = {
        "model": model,
        "lr": args.lr,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_cycles * args.n_epochs_per_cycle,
        "parserargs": args,
    }

    return ssl_type.initialize(**ssl_args)


def run(
    args: Namespace,
    seed: int = 42,
    run_idx: int = 0,
) -> dict:
    set_seed(seed)

    args.train_batch_size = args.train_batch_size // torch.cuda.device_count()

    checkpoint_filename = (
        args.experiment_name + "_" + args.imagenet_variant + "_" + str(datetime.now())
    )

    dataset_pickle_filename = args.imagenet_variant + "_" + args.imbalance_method

    if not args.calc_novelty_score and args.pretrain:
        datamodule = init_datamodule(
            args,
            dataset_pickle_filename,
        )

    else:
        datamodule = None

    model = init_model(args)

    ssl_type = init_ssl_type(args, model)

    if args.calc_novelty_score:
        return calc_novelty_score(args, ssl_type)

    if args.checkpoint is not None:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        # is backbone inside any key string?
        if sum(["backbone" in key for key in state_dict.keys()]):
            # Create a new state dict with renamed keys
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("backbone", "model.resnet")
                new_state_dict[new_key] = state_dict[key]

            state_dict = new_state_dict

        if sum(["module" in key for key in state_dict.keys()]):
            # Create a new state dict with renamed keys
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("module", "model.resnet")
                new_state_dict[new_key] = state_dict[key]

            state_dict = new_state_dict

        missing, unexpected = ssl_type.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    checkpoints_dir = os.environ["BASE_CACHE_DIR"] + "/checkpoints"

    os.makedirs(checkpoints_dir, exist_ok=True)

    last_epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=checkpoint_filename + "-last-epoch-{epoch}-{val_loss:.4f}",
        save_top_k=0,  # Setting this to 0 disables saving based on a metric
    )

    every_20_epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=checkpoint_filename + "-epoch-{epoch}-{val_loss:.4f}",
        every_n_epochs=100,  # Save every 20 epochs
        save_top_k=-1,  # This allows saving all checkpoints matching every_n_epochs
    )

    if args.logger and not args.test_mode:
        log_name = args.experiment_name if args.experiment_name else checkpoint_filename
        os.environ["WANDB_DIR"] = os.environ["BASE_CACHE_DIR"]
        wandb_logger = WandbLogger(
            project="FOMO2",
            name=log_name + str(seed),
            group=log_name,
            save_dir=os.environ["BASE_CACHE_DIR"],
        )
        # wandb_logger.watch(model, log="all")

    stats_monitor = DeviceStatsMonitor()

    callbacks = [last_epoch_checkpoint, every_20_epoch_checkpoint, stats_monitor]

    trainer_args = {
        "max_epochs": args.n_epochs_per_cycle,
        "accumulate_grad_batches": args.grad_acc_steps,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger if args.logger and not args.test_mode else None,
    }

    if torch.cuda.is_available():
        strategy = DeepSpeedStrategy(
            config={
                "train_batch_size": args.train_batch_size
                * args.grad_acc_steps
                * torch.cuda.device_count(),
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "offload_param": {"device": "cpu", "pin_memory": True},
                },
            },
        )
        os.environ["DEEPSPEED_COMMUNICATION_CLIENT_WAIT_TIMEOUT"] = "7200"
        trainer_args.update(
            {
                "strategy": strategy,
                "precision": "bf16",
                "devices": "auto",  # Let PyTorch Lightning handle device selection
                "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"],
            }
        )
        trainer_args.pop("accelerator", None)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs Available: ", torch.cuda.device_count())

    imbalanced_training = ImbalancedTraining(
        args,
        trainer_args,
        ssl_type,
        datamodule,
        checkpoint_filename=checkpoint_filename,
        checkpoint_callback=callbacks[0],
        run_idx=run_idx,
    )

    results = imbalanced_training.run()

    if args.logger and not args.test_mode:
        wandb_logger.experiment.unwatch()

    return results


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

        results = run(
            run_args,
            seed=args.seeds[run_idx],
            run_idx=run_idx,
        )

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({"training_time": training_time})

        print(results)

        all_results.append(results)

    return all_results


def main():

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    args = get_training_args()

    # add a timestamp to the additional data path
    args.additional_data_path = (
        os.environ["BASE_CACHE_DIR"]
        + "/"
        + args.additional_data_path
        + "_"
        + generate_random_string()
    )

    if not args.test_mode:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    all_results = run_different_seeds(args)

    print_mean_std(all_results)


if __name__ == "__main__":
    main()
