import argparse
from experiment.models.ModelTypes import ModelTypes
from experiment.models.SSLTypes import SSLTypes
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


def get_training_args(get_defaults: bool = False) -> dict:
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

    parser.add_argument(
        "--finetuning_benchmarks",
        nargs="+",
        type=str,
        choices=FinetuningBenchmarks.get_all_benchmark_names(),
        default=FinetuningBenchmarks.get_default_benchmark_names(),
    )

    parser.add_argument(
        "--test_mode",
        action="store_true",
    )
    parser.add_argument(
        "--no-test_mode",
        action="store_false",
        dest="test_mode",
    )
    parser.set_defaults(test_mode=False)

    # Use augmentation at all, or just train on the original dataset forever
    parser.add_argument("--ood_augmentation", action="store_true")
    parser.set_defaults(ood_augmentation=False)

    parser.add_argument("--remove_diffusion", action="store_true")
    parser.set_defaults(remove_ood_detection=False)

    # Use OOD-detection or uniformly augment the dataset
    parser.add_argument("--use_ood", action="store_true")
    parser.add_argument("--no-use_ood", action="store_false", dest="use_ood")
    parser.set_defaults(use_ood=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_cycles", type=int, default=20)
    parser.add_argument("--n_epochs_per_cycle", type=int, default=20)

    parser.add_argument("--splits", nargs="+", type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument(
        "--batch_size", type=int, default=128
    )  # this should be the simclr batch size but ofxourse depends on gpu a bit
    parser.add_argument("--checkpoint", nargs="+", type=str, default=None)

    parser.add_argument("--no-pretrain", action="store_false", dest="pretrain")
    parser.set_defaults(pretrain=True)

    parser.add_argument("--no-finetune", action="store_false", dest="finetune")
    parser.set_defaults(finetune=True)

    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--max_hours_per_run", type=int, default=5)

    parser.set_defaults(logger=True)
    parser.add_argument("--no-logger", action="store_false", dest="logger")

    parser.add_argument("--classification_head", action="store_true")

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    # I-Jepa Args
    parser.add_argument("--fe_batch_size", type=int, default=32)
    parser.add_argument("--sd_batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--pct_ood", type=float, default=0.15)
    parser.add_argument("--pct_train", type=float, default=1.0)
    parser.add_argument("--ood_test_split", type=float, default=0.1)
    parser.add_argument("--additional_data_path", type=str, default="additional_data")

    # I-Jepa Args

    # Data
    parser.add_argument("--color_jitter_strength", type=float, default=0.0)
    parser.add_argument("--crop_scale", type=float, nargs=2, default=[0.3, 1.0])
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument(
        "--image_folder", type=str, default="imagenet_full_size/061417/"
    )
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
    parser.add_argument("--pred_depth", type=int, default=3)
    parser.add_argument("--pred_emb_dim", type=int, default=192)
    parser.add_argument("--use_bfloat16", type=bool, default=False)

    # Optimization
    parser.add_argument("--ema", type=float, nargs=2, default=[0.996, 1.0])
    parser.add_argument("--final_lr", type=float, default=1.0e-4)
    parser.add_argument("--final_weight_decay", type=float, default=0.4)
    parser.add_argument("--ipe_scale", type=float, default=1.0)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--start_lr", type=float, default=0.0002)
    parser.add_argument(
        "--warmup", type=int, default=5
    )  # n_batches, so change depending on the gpu, but if the batch size goes up maybe 1 epoch longer warmup isnt too bad
    # parser.add_argument('--weight_decay', type=float, default=0.04)

    if get_defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
