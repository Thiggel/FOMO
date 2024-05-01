from datetime import date
import time
import argparse
import lightning as L
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        DeviceStatsMonitor
    )
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.multiprocessing as mp

from utils.set_seed import set_seed
from dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from models.ModelTypes import ModelTypes
from models.SSLTypes import SSLTypes
from models.FinetuningBenchmarks import FinetuningBenchmarks
from dataset.ImageNetVariants import ImageNetVariants


mp.set_start_method('spawn')


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dataset/novel')

    parser.add_argument(
        '--model_name',
        type=str,
        choices=ModelTypes.get_model_types(),
        default=ModelTypes.get_default_model_type()
    )

    parser.add_argument(
        '--imagenet_variant',
        type=str,
        choices=ImageNetVariants.get_variants(),
        default=ImageNetVariants.get_default_variant()
    )

    parser.add_argument(
        '--ssl_method',
        type=str,
        choices=SSLTypes.get_ssl_types(),
        default=SSLTypes.get_default_ssl_type()
    )

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_epochs', type=int, default=500)

    parser.add_argument('--splits', nargs = '+', type = float, default=[0.8, 0.1, 0.1])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--no-pretrain', action='store_false', dest='pretrain')
    parser.set_defaults(pretrain=True)

    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no-finetune', action='store_false', dest='finetune')
    parser.set_defaults(finetune=True)

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--max_hours_per_run', type=int, default=5)

    parser.add_argument('--logger', action='store_true')
    parser.add_argument('--no-logger', action='store_false', dest='logger')
    parser.add_argument('--csv_file', type=str, default='default.csv',
                    help='CSV file name')
    parser.set_defaults(logger=True)

    args = parser.parse_args()
    args.splits = tuple(args.splits)

    return args


def get_model_name(module: L.LightningModule) -> str:
    if not hasattr(module, 'model'):
        return module.__class__.__name__

    return module.model.__class__.__name__


def finetune(
    args: dict,
    trainer_args: dict,
    model: L.LightningModule,
) -> dict:
    benchmarks = FinetuningBenchmarks.benchmarks
    results = {}

    for benchmark in benchmarks:
        finetuner = benchmark(
            model=model.model,
            lr=args.lr,
        )
        trainer = L.Trainer(
            **trainer_args
        )

        trainer.fit(model=finetuner)

        results.update(trainer.test(model=finetuner))

    return results


def run(args: dict, seed: int = 42) -> dict:
    set_seed(seed)

    model_type = ModelTypes.get_model_type(args.model_name)

    datamodule = ImbalancedImageNetDataModule(
        dataset_variant=ImageNetVariants.init_variant(args.imagenet_variant),
        splits=args.splits,
        batch_size=args.batch_size,
        resized_image_size=model_type.resized_image_size,
    )

    #I know this is terrible please help me fix it
    #this needs to be removed for some reason 'model_name': args.model_name,
    #'resized_image_size': model_type.resized_image_size,
    #    'batch_size': args.batch_size,
    model_args = {
        'output_size': int(args.imagenet_variant), 
    }

    model = model_type.initialize(**model_args)

    ssl_type = SSLTypes.get_ssl_type(args.ssl_method)
    ssl_args = {
        'model': model,
        'lr': args.lr,
        'temperature': args.temperature,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
    }
    ssl_method = ssl_type.initialize(**ssl_args)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=args.model_name + ' ' + args.root_dir + '-{epoch}-{val_loss:.2f}',
        monitor='val_loss',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min',
    )

    tensorboard_logger = TensorBoardLogger(
        'logs/',
        name=args.model_name + ' ' + args.csv_file
    )

    if args.checkpoint is not None:
        model.load_state_dict(
            torch.load(
                args.checkpoint,
                map_location=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            )['state_dict']
        )

    stats_monitor = DeviceStatsMonitor()

    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        stats_monitor
    ]

    trainer_args = {
        'max_time': {'hours': args.max_hours_per_run},
        'max_epochs': args.max_epochs,
        'callbacks': callbacks,
        'enable_checkpointing': True,
        'logger': tensorboard_logger if args.logger else None,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 'auto',
    }

    trainer = L.Trainer(
        **trainer_args
    )

    if args.pretrain:
        trainer.fit(model=ssl_method, datamodule=datamodule)

        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)['state_dict']
        )

    if args.finetune:
        results = finetune(args, trainer_args, model)

        return results

    else:
        return {}


def print_mean_std(results: list[dict]) -> None:
    tensor_data = torch.tensor([list(d.values()) for d in results])

    mean = tensor_data.mean(dim=0)
    std_dev = tensor_data.std(dim=0)

    for key, m, s in zip(results[0].keys(), mean, std_dev):
        print(f'{key} - Mean: {m.item()}, Std: {s.item()}')


if __name__ == '__main__':
    args = get_args()

    all_results = []

    for run_idx in range(args.num_runs):
        start_time = time.time()

        results = run(args, seed=run_idx)[0]

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({'training_time': training_time})

        print(results)

        all_results.append(results)

    print_mean_std(all_results)
