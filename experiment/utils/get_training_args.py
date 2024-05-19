import argparse
from experiment.models.ModelTypes import ModelTypes
from experiment.models.SSLTypes import SSLTypes
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


def get_training_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=ModelTypes.get_model_types(),
        default=ModelTypes.get_default_model_type(),
    )

    parser.add_argument(
        "--imagenet_variant",
        type=str,
        choices=ImageNetVariants.get_variants(),
        default=ImageNetVariants.get_default_variant(),
    )

    parser.add_argument(
        "--imbalance_method",
        type=str,
        choices=ImbalanceMethods.get_methods(),
        default=ImbalanceMethods.get_default_method(),
    )

    parser.add_argument(
        "--ssl_method",
        type=str,
        choices=SSLTypes.get_ssl_types(),
        default=SSLTypes.get_default_ssl_type(),
    )

    parser.add_argument("--no_augmentation", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_cycles", type=int, default=20)
    parser.add_argument("--n_epochs_per_cycle", type=int, default=20)

    parser.add_argument("--splits", nargs="+", type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--checkpoint", nargs="+", type=str, default=None)

    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--no-pretrain", action="store_false", dest="pretrain")
    parser.set_defaults(pretrain=True)

    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--no-finetune", action="store_false", dest="finetune")
    parser.set_defaults(finetune=True)

    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_hours_per_run", type=int, default=5)

    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--no-logger", action="store_false", dest="logger")
    parser.set_defaults(logger=True)

    parser.add_argument("--classification_head", action="store_true")
    parser.add_argument("--early_stopping_monitor", type=str, default="val_loss")

    #I-Jepa Args
    parser.add_argument("--fe_batch_size", type=int, default=32)
    parser.add_argument("--sd_batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--pct_ood", type=float, default=0.1)
    parser.add_argument("--pct_train", type=float, default=1.0)
    parser.add_argument("--ood_test_split", type=float, default=0.1)
    parser.add_argument("--additional_data_path", type=str, default="additional_data")

    # I-Jepa Args

    # Data
    parser.add_argument("--color_jitter_strength", type=float, default=0.0)
    parser.add_argument("--crop_scale", type=float, nargs=2, default=[0.3, 1.0])
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument(
        "--image_folder", type=str, default="imagenet_full_size/061417/"
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", type=bool, default=True)
    # parser.add_argument('--root_path', type=str, default='$replace_this_with_absolute_path_to_your_datasets_directory') #idk if I can make this nicer somehow.
    parser.add_argument("--use_color_distortion", type=bool, default=False)
    parser.add_argument("--use_gaussian_blur", type=bool, default=False)
    parser.add_argument("--use_horizontal_flip", type=bool, default=False)

    # Mask
    parser.add_argument("--allow_overlap", type=bool, default=False)
    parser.add_argument("--aspect_ratio", type=float, nargs=2, default=[0.75, 1.5])
    parser.add_argument("--enc_mask_scale", type=float, nargs=2, default=[0.85, 1.0])
    parser.add_argument("--min_keep", type=int, default=10)
    parser.add_argument("--num_enc_masks", type=int, default=1)
    parser.add_argument("--num_pred_masks", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--pred_mask_scale", type=float, nargs=2, default=[0.15, 0.2])

    # Meta
    parser.add_argument("--pred_depth", type=int, default=12)
    parser.add_argument("--pred_emb_dim", type=int, default=384)
    parser.add_argument("--use_bfloat16", type=bool, default=False)

    # Optimization
    parser.add_argument("--ema", type=float, nargs=2, default=[0.996, 1.0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--final_lr", type=float, default=1.0e-06)
    parser.add_argument("--final_weight_decay", type=float, default=0.4)
    parser.add_argument("--ipe_scale", type=float, default=1.0)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--start_lr", type=float, default=0.0002)
    parser.add_argument("--warmup", type=int, default=40)
    # parser.add_argument('--weight_decay', type=float, default=0.04)

    args = parser.parse_args()

    args.split = tuple(args.splits)

    return args
