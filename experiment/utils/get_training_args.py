import argparse
from experiment.models.ModelTypes import ModelTypes
from experiment.models.SSLTypes import SSLTypes
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import FinetuningBenchmarks
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


def get_training_args() -> dict:
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
        '--imbalance_method',
        type=str,
        choices=ImbalanceMethods.get_methods(),
        default=ImbalanceMethods.get_default_method()
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
    parser.add_argument('--max_cycles', type=int, default=20)
    parser.add_argument('--n_epochs_per_cycle', type=int, default=20)

    parser.add_argument('--splits', nargs='+', type=float, default=[0.8, 0.1, 0.1])
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
    parser.set_defaults(logger=True)

    args = parser.parse_args()

    args.split = tuple(args.splits)

    return args

