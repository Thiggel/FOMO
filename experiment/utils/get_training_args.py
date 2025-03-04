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

    # Core experiment settings
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=ModelTypes.get_model_types(),
        default=ModelTypes.get_default_model_type(),
        help="Model architecture",
    )
    parser.add_argument(
        "--imagenet-variant",
        type=str,
        choices=ImageNetVariants.get_variants(),
        default=ImageNetVariants.get_default_variant(),
        help="ImageNet dataset variant",
    )
    parser.add_argument(
        "--imbalance-method",
        type=str,
        choices=ImbalanceMethods.get_methods(),
        default=ImbalanceMethods.get_default_method(),
        help="Method for creating dataset imbalance",
    )
    parser.add_argument(
        "--ssl-method",
        type=str,
        choices=SSLTypes.get_ssl_types(),
        default=SSLTypes.get_default_ssl_type(),
        help="Self-supervised learning method",
    )

    # Finetuning settings
    parser.add_argument(
        "--finetuning-benchmarks",
        nargs="+",
        type=str,
        choices=FinetuningBenchmarks.get_all_benchmark_names(),
        default=FinetuningBenchmarks.get_default_benchmark_names(),
        help="Finetuning benchmark datasets",
    )

    # Model architecture settings
    parser.add_argument(
        "--classification-head",
        action="store_true",
        help="Use classification head in model",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for ViT models",
    )

    # Training configuration
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive learning",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="Number of training cycles",
    )
    parser.add_argument(
        "--n-epochs-per-cycle",
        type=int,
        default=20,
        help="Epochs per cycle",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup epochs",
    )

    # Batch size and processing
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=512,
        help="Validation batch size",
    )
    parser.add_argument(
        "--sd-batch-size",
        type=int,
        default=4,
        help="Stable diffusion batch size",
    )

    # Data processing settings
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Image crop size",
    )

    # Dataset and splitting
    parser.add_argument(
        "--splits",
        nargs="+",
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Dataset splits",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset",
    )

    # OOD and augmentation settings
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="k for kNN computation (number of neighbors to consider)",
    )
    parser.add_argument(
        "--num-ood-samples",
        type=int,
        default=1000,
        help="Select the N most OOD samples in the dataset for augmentation after each cycle",
    )
    parser.add_argument(
        "--num-generations-per-ood-sample",
        type=int,
        default=100,
        help="Number of augmentations to generate per OOD image",
    )
    parser.add_argument(
        "--save-class-distribution",
        action="store_true",
        help="Save class distribution of dataset",
    )

    # Masking settings for SSL
    parser.add_argument(
        "--allow-overlap",
        type=bool,
        default=False,
        help="Allow overlap in masking",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        nargs=2,
        default=[0.75, 1.5],
        help="Aspect ratio range for masking",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        choices=["stable_diffusion", "flux"],
        default="stable_diffusion",
    )
    parser.add_argument(
        "--enc-mask-scale",
        type=float,
        nargs=2,
        default=[0.85, 1.0],
        help="Encoder mask scale range",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=10,
        help="Minimum tokens to keep",
    )
    parser.add_argument(
        "--num-enc-masks",
        type=int,
        default=1,
        help="Number of encoder masks",
    )
    parser.add_argument(
        "--num-pred-masks",
        type=int,
        default=4,
        help="Number of predictor masks",
    )
    parser.add_argument(
        "--pred-mask-scale",
        type=float,
        nargs=2,
        default=[0.15, 0.2],
        help="Predictor mask scale range",
    )

    # Model and optimization parameters
    parser.add_argument(
        "--pred-depth",
        type=int,
        default=3,
        help="Prediction head depth",
    )
    parser.add_argument(
        "--pred-emb-dim",
        type=int,
        default=192,
        help="Prediction embedding dimension",
    )
    parser.add_argument(
        "--ipe-scale",
        type=float,
        default=1.0,
        help="IPE scale factor",
    )
    parser.add_argument(
        "--use-bfloat16",
        type=bool,
        default=False,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--ema",
        type=float,
        nargs=2,
        default=[0.996, 1.0],
        help="EMA decay range",
    )

    # Run configuration
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of training runs",
    )
    parser.add_argument(
        "--max-hours-per-run",
        type=int,
        default=12,
        help="Max hours per run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        type=str,
        default=None,
        help="Path to checkpoint(s)",
    )

    # Feature extraction and evaluation
    parser.add_argument(
        "--calc-novelty-score",
        action="store_true",
        help="Calculate novelty scores",
    )

    # Flags
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--no-logger",
        action="store_false",
        dest="logger",
        help="Disable logging",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_false",
        dest="pretrain",
        help="Skip pretraining",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_false",
        dest="finetune",
        help="Skip finetuning",
    )
    parser.add_argument(
        "--no-use-ood", action="store_false", dest="use_ood", help="Use OOD detection"
    )
    parser.add_argument(
        "--remove-diffusion",
        action="store_true",
        help="Remove diffusion-based augmentation",
    )
    parser.add_argument(
        "--ood-augmentation",
        action="store_true",
        help="Enable OOD augmentation",
    )

    # Set defaults for flags
    parser.set_defaults(
        test_mode=False,
        logger=True,
        pretrain=True,
        finetune=True,
        use_ood=True,
        remove_diffusion=False,
        ood_augmentation=False,
        classification_head=False,
        calc_novelty_score=False,
        save_class_distribution=False,
        streaming=False,
    )

    if get_defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
