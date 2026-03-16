from datetime import datetime
import copy
import torch
import os
import json
import re

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
import hydra
from omegaconf import DictConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def _sanitize_path_token(token: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")
    return sanitized or "run"


def resolve_additional_data_path(args: DictConfig) -> str:
    configured_path = args.get("additional_data_path")
    if configured_path:
        return str(configured_path)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    shared_token = os.environ.get("TORCHELASTIC_RUN_ID")
    if shared_token and shared_token.lower() != "none":
        token = _sanitize_path_token(shared_token)
    elif os.environ.get("SLURM_JOB_ID"):
        job_id = os.environ["SLURM_JOB_ID"]
        step_id = os.environ.get("SLURM_STEP_ID")
        token = _sanitize_path_token(
            f"slurm_{job_id}" + (f"_{step_id}" if step_id else "")
        )
    elif world_size > 1:
        # Keep a stable token across ranks when torchrun/slurm IDs are unavailable.
        master_port = os.environ.get("MASTER_PORT", "0")
        token = _sanitize_path_token(f"launch_{os.getppid()}_{master_port}")
    else:
        token = generate_random_string()

    return os.path.join(os.environ["BASE_CACHE_DIR"], f"additional_data_{token}")


def checkpoint_seed_dir(args: DictConfig, seed: int) -> str:
    dataset_id = args.dataset.dataset_path.replace("/", "_")
    experiment_id = args.experiment_name if args.experiment_name else "default_experiment"
    checkpoint_root_dir = os.environ.get(
        "CHECKPOINT_ROOT_DIR",
        os.path.join(os.environ["BASE_CACHE_DIR"], "checkpoints"),
    )
    return os.path.join(
        checkpoint_root_dir,
        experiment_id,
        dataset_id,
        f"seed_{seed}",
    )


def configured_seed_list(args: DictConfig) -> list[int]:
    num_runs = int(args.num_runs)
    if num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")

    seeds = [int(seed) for seed in list(args.seeds)]
    if len(seeds) < num_runs:
        raise ValueError(
            f"Not enough seeds configured: need {num_runs}, got {len(seeds)}"
        )

    return seeds[:num_runs]


def resolve_seed_schedule(args: DictConfig) -> list[tuple[int, int]]:
    configured_seeds = configured_seed_list(args)
    all_seed_values = [int(seed) for seed in list(args.seeds)]

    explicit_seed = args.get("seed")
    if explicit_seed is not None:
        seed = int(explicit_seed)
        run_idx = all_seed_values.index(seed) if seed in all_seed_values else 0
        return [(run_idx, seed)]

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is not None:
        task_idx = int(slurm_task_id)
        if task_idx < 0 or task_idx >= len(configured_seeds):
            raise ValueError(
                f"SLURM_ARRAY_TASK_ID {task_idx} is out of range for {len(configured_seeds)} configured seeds"
            )
        return [(task_idx, configured_seeds[task_idx])]

    return list(enumerate(configured_seeds))


def resolve_training_schedule(args: DictConfig) -> tuple[int, int, int]:
    """Derive the training schedule from CLI overrides and defaults."""

    num_cycles = args.get("max_cycles")
    num_cycles = num_cycles if num_cycles is not None else args.num_cycles

    if num_cycles <= 0:
        raise ValueError("num_cycles must be a positive integer")

    epochs_per_cycle_override = args.get("n_epochs_per_cycle")
    if epochs_per_cycle_override is not None:
        total_epochs = epochs_per_cycle_override * num_cycles
        epochs_per_cycle = epochs_per_cycle_override
    else:
        total_epochs = args.total_epochs
        if total_epochs < num_cycles:
            raise ValueError("total_epochs must be >= num_cycles")
        if total_epochs % num_cycles != 0:
            raise ValueError("total_epochs must be divisible by num_cycles")
        epochs_per_cycle = total_epochs // num_cycles

    return num_cycles, total_epochs, epochs_per_cycle


def init_datamodule(args: DictConfig, checkpoint_filename: str) -> L.LightningDataModule:
    ssl_method = SSLTypes.get_ssl_type(args.ssl.ssl_method)

    return ImbalancedDataModule(
        collate_fn=ssl_method.collate_fn(args),
        dataset_path=args.dataset.dataset_path,
        dataset_name=args.dataset.get("dataset_name"),
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
        "weight_decay": args.ssl.weight_decay,
        "max_epochs": args.total_epochs,
        "parserargs": args,
        "use_temperature_schedule": args.use_temperature_schedule,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "t_max": args.t_max,
        "sdclr_prune_rate": args.get("sdclr_prune_rate", 0.0),
    }
    if "temperature" in args.ssl:
        ssl_args["temperature"] = args.ssl.temperature

    return ssl_type.initialize(**ssl_args)


def run(
    args: DictConfig,
    seed: int = 42,
    run_idx: int = 0,
) -> dict:
    set_seed(seed)

    # Resolve potential CLI overrides for cycle configuration before anything else
    num_cycles, total_epochs, epochs_per_cycle = resolve_training_schedule(args)
    args.num_cycles = num_cycles
    args.total_epochs = total_epochs
    args.n_epochs_per_cycle = epochs_per_cycle

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.train_batch_size % world_size != 0:
        raise ValueError(
            f"train_batch_size ({args.train_batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )
    args.train_batch_size = args.train_batch_size // world_size

    dataset_id = args.dataset.dataset_path.replace("/", "_")
    experiment_id = args.experiment_name if args.experiment_name else "default_experiment"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    checkpoint_filename = f"{experiment_id}_{dataset_id}_{timestamp}"

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

    checkpoints_dir = checkpoint_seed_dir(args, seed)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if args.checkpoint is not None:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, weights_only=False)
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

    has_pretrain_validation = (
        datamodule is not None and len(datamodule.val_dataset) > 0
    )
    last_epoch_checkpoint_filename = (
        checkpoint_filename + "-last-epoch-{epoch}-{val_loss:.4f}"
        if has_pretrain_validation
        else checkpoint_filename + "-last-epoch-{epoch}"
    )
    epoch_checkpoint_filename = (
        checkpoint_filename + "-epoch-{epoch}-{val_loss:.4f}"
        if has_pretrain_validation
        else checkpoint_filename + "-epoch-{epoch}"
    )

    last_epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=last_epoch_checkpoint_filename,
        save_top_k=0,  # Setting this to 0 disables saving based on a metric
        save_last=True,
        enable_version_counter=False,
    )

    if args.logger:
        log_name = args.experiment_name if args.experiment_name else checkpoint_filename
        wandb_save_dir = os.environ.get("WANDB_DIR", os.environ["BASE_CACHE_DIR"])
        os.makedirs(wandb_save_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_save_dir
        wandb_logger = WandbLogger(
            project="FOMO5",
            name=log_name + str(seed),
            group=log_name,
            save_dir=wandb_save_dir,
            settings=wandb.Settings(silent=True),
        )
        # wandb_logger.watch(model, log="all")

        print("CHECKPOINT FILENAME: ", checkpoint_filename)

    stats_monitor = DeviceStatsMonitor()

    callbacks = [last_epoch_checkpoint]
    if os.environ.get("FOMO_SAVE_EPOCH_CHECKPOINTS", "0") == "1":
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename=epoch_checkpoint_filename,
                every_n_epochs=int(os.environ.get("FOMO_EVERY_N_EPOCHS", "200")),
                save_top_k=-1,
                enable_version_counter=False,
            )
        )
    callbacks.append(stats_monitor)

    trainer_args = {
        "max_epochs": epochs_per_cycle,
        "accumulate_grad_batches": args.grad_acc_steps,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger if args.logger else None,
    }

    if torch.cuda.is_available():
        lightning_devices = world_size if world_size > 1 else 1
        trainer_args.update(
            {
                "accelerator": "cuda",
                "devices": lightning_devices,
                "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"],
            }
        )
        if world_size > 1:
            ssl_method_name = str(args.ssl.ssl_method).lower()
            if ssl_method_name == "moco":
                trainer_args["strategy"] = "ddp_find_unused_parameters_true"
            else:
                trainer_args["strategy"] = "ddp"

        print(
            f"[RANK {global_rank}/{world_size} | LOCAL_RANK {local_rank}] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch.cuda.device_count()={torch.cuda.device_count()} "
            f"per_rank_train_batch_size={args.train_batch_size}"
        )

    imbalanced_training = ImbalancedTraining(
        args,
        trainer_args,
        ssl_type,
        datamodule,
        checkpoint_filename=checkpoint_filename,
        checkpoint_callback=last_epoch_checkpoint,
        run_idx=run_idx,
    )

    results = imbalanced_training.run()

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results


def set_checkpoint_for_run(args: DictConfig, run_idx: int) -> DictConfig:
    checkpoint_list = args.checkpoint_list
    if checkpoint_list is None:
        checkpoint_list = args.checkpoint

    if checkpoint_list is None:
        args.checkpoint = None
        return args

    if isinstance(checkpoint_list, str):
        checkpoints = [checkpoint_list]
    else:
        checkpoints = list(checkpoint_list)

    if len(checkpoints) == 0:
        args.checkpoint = None
        return args

    args.checkpoint = checkpoints[run_idx % len(checkpoints)]

    return args


def run_different_seeds(args: DictConfig) -> list[dict]:
    all_results = []
    seed_schedule = resolve_seed_schedule(args)
    use_seed_specific_data_path = len(seed_schedule) > 1

    for run_idx, seed in seed_schedule:
        start_time = time.time()
        seed_dir = checkpoint_seed_dir(args, seed)
        os.makedirs(seed_dir, exist_ok=True)
        seed_result_file = os.path.join(seed_dir, "result.json")

        run_args = set_checkpoint_for_run(copy.deepcopy(args), run_idx)
        if use_seed_specific_data_path:
            run_args.additional_data_path = f"{args.additional_data_path}_seed_{seed}"

        results = run(
            run_args,
            seed=seed,
            run_idx=run_idx,
        )

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({"training_time": training_time})

        print(results)

        with open(seed_result_file, "w") as f:
            json.dump(results, f)

        all_results.append(results)

    return all_results


def aggregate_seed_results(args: DictConfig) -> list[dict]:
    all_results = []
    missing_files = []

    for seed in configured_seed_list(args):
        seed_result_file = os.path.join(checkpoint_seed_dir(args, seed), "result.json")
        if not os.path.exists(seed_result_file):
            missing_files.append(seed_result_file)
            continue

        with open(seed_result_file, "r") as f:
            all_results.append(json.load(f))

    if missing_files:
        raise FileNotFoundError(
            "Missing seed result files:\n" + "\n".join(missing_files)
        )

    return all_results


def run_app(args: DictConfig) -> None:
    load_dotenv()
    # login(token=os.getenv("HUGGINGFACE_TOKEN"))

    if bool(args.get("aggregate_only", False)):
        print_mean_std(aggregate_seed_results(args))
        return

    if not bool(args.get("enable_media_logging", False)):
        args.log_tsne = False
        args.log_class_dist = False
        args.log_generated_samples = False
        args.save_visualization_data = False

    args.additional_data_path = resolve_additional_data_path(args)

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
