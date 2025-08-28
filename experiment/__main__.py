from datetime import datetime
import torch
import os

try:
    from lightning.pytorch.strategies import DeepSpeedStrategy
except Exception:  # pragma: no cover - deepspeed optional
    DeepSpeedStrategy = None

from dotenv import load_dotenv
import time
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch import nn
import torch.multiprocessing as mp
from huggingface_hub import login
import hydra
from omegaconf import DictConfig

from experiment.utils.set_seed import set_seed
from experiment.utils.print_mean_std import print_mean_std
from experiment.utils.get_model_name import get_model_name
from experiment.utils.generate_random_string import generate_random_string
from experiment.utils.calc_novelty_score import calc_novelty_score

from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.models.ModelTypes import ModelTypes
from experiment.models.SSLTypes import SSLTypes
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.ImbalancedTraining import ImbalancedTraining

mp.set_start_method("spawn")
torch.multiprocessing.set_sharing_strategy("file_system")
import shutil
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def init_datamodule(args: DictConfig, checkpoint_filename: str) -> L.LightningDataModule:
    ssl_method = SSLTypes.get_ssl_type(args.ssl.ssl_method)

    return ImbalancedDataModule(
        collate_fn=ssl_method.collate_fn(args),
        dataset_path=args.dataset.dataset_path,
        split=args.dataset.split,
        x_key=args.dataset.x_key,
        y_key=args.dataset.y_key,
        imbalance_method=ImbalanceMethods.init_method(args.dataset.imbalance_method),
        splits=args.splits,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        checkpoint_filename=checkpoint_filename,
        transform=ssl_method.transforms(args),
        additional_data_path=args.additional_data_path,
    )


def init_model(args: DictConfig) -> nn.Module:
    model_type = ModelTypes.get_model_type(args.model.model_name)

    model_args = {
        "model_name": args.model.model_name,
        "batch_size": args.train_batch_size,
        "output_size": 128,  # simclear uses this hidden dim, vit doesnt use this parameter
        "image_size": args.crop_size,
        "classification_head": args.model.classification_head,
    }

    model = model_type.initialize(**model_args)

    return model


def init_ssl_type(
    args: DictConfig,
    model: nn.Module,
) -> L.LightningModule:
    ssl_type = SSLTypes.get_ssl_type(args.ssl.ssl_method)
    ssl_args = {
        "model": model,
        "lr": args.ssl.lr,
        "temperature": args.ssl.temperature,
        "weight_decay": args.ssl.weight_decay,
        "max_epochs": args.max_cycles * args.n_epochs_per_cycle,
        "parserargs": args,
        "use_temperature_schedule": args.use_temperature_schedule,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "t_max": args.t_max,
        "sdclr_prune_rate": args.get("sdclr_prune_rate", 0.0),
    }

    return ssl_type.initialize(**ssl_args)


def run(
    args: DictConfig,
    seed: int = 42,
    run_idx: int = 0,
) -> dict:
    set_seed(seed)

    args.train_batch_size = args.train_batch_size // torch.cuda.device_count()

    dataset_id = args.dataset.dataset_path.replace("/", "_")
    checkpoint_filename = (
        args.experiment_name + "_" + dataset_id + "_" + str(datetime.now())
    )

    dataset_pickle_filename = dataset_id + "_" + args.dataset.imbalance_method

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
        every_n_epochs=200,  # Save every 20 epochs
        save_top_k=-1,  # This allows saving all checkpoints matching every_n_epochs
    )

    if args.logger:
        log_name = args.experiment_name if args.experiment_name else checkpoint_filename
        os.environ["WANDB_DIR"] = os.environ["BASE_CACHE_DIR"]
        wandb_logger = WandbLogger(
            project="FOMO4",
            name=log_name + str(seed),
            group=log_name,
            save_dir=os.environ["BASE_CACHE_DIR"],
            settings=wandb.Settings(silent=True),
        )
        # wandb_logger.watch(model, log="all")

        print("CHECKPOINT FILENAME: ", checkpoint_filename)

    stats_monitor = DeviceStatsMonitor()

    callbacks = [last_epoch_checkpoint, every_20_epoch_checkpoint, stats_monitor]

    trainer_args = {
        "max_epochs": args.n_epochs_per_cycle,
        "accumulate_grad_batches": args.grad_acc_steps,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger if args.logger else None,
    }

    if torch.cuda.is_available():
        if args.use_deepspeed and DeepSpeedStrategy is not None:
            strategy = DeepSpeedStrategy(
                config={
                    "train_batch_size": args.train_batch_size,
                    "zero_optimization": {"stage": 2},
                    "zero_allow_untested_optimizer": True,
                },
            )
            os.environ["DEEPSPEED_COMMUNICATION_CLIENT_WAIT_TIMEOUT"] = "7200"
            trainer_args.update(
                {
                    "strategy": strategy,
                    "devices": "auto",  # Let PyTorch Lightning handle device selection
                    "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"],
                }
            )
            trainer_args.pop("accelerator", None)
        else:
            trainer_args.update(
                {
                    "accelerator": "cuda",
                    "devices": "auto",
                    "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"],
                }
            )

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

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results


def set_checkpoint_for_run(args: DictConfig, run_idx: int) -> str:
    if not hasattr(args, "checkpoint_list") or args.checkpoint_list is None:
        args.checkpoint_list = args.checkpoint

    if args.checkpoint_list is not None:
        args.checkpoint = args.checkpoint_list[run_idx % len(args.checkpoint_list)]

    return args


def run_different_seeds(args: DictConfig) -> dict:
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


def run_app(args: DictConfig) -> None:
    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    args.additional_data_path = (
        os.environ["BASE_CACHE_DIR"]
        + "/"
        + "additional_data"
        + "_"
        + generate_random_string()
    )

    if args.logger:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    all_results = run_different_seeds(args)

    print_mean_std(all_results)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_app(cfg)


if __name__ == "__main__":
    main()
