#!/bin/bash
#SBATCH --job-name=fomo-smoke
#SBATCH --partition=gpu-a100:test
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/usr/bemflait/FOMO_runtime/logs/smoke-%j.out

set -euo pipefail
: "${WORK:?WORK must be set on Lise}"
cd "$WORK/FOMO"
. jobs/clusters/lise_environment.sh
export FOMO_ALLOW_DATA_DOWNLOAD=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m experiment \
    model=resnet18 ssl=simclr dataset=cifar10_imbalanced \
    logger=false pretrain=true finetune=false \
    max_cycles=2 n_epochs_per_cycle=1 \
    train_batch_size=16 val_batch_size=16 \
    limit_train_batches=1 limit_val_batches=1 num_sanity_val_steps=0 \
    num_ood_samples=2 num_generations_per_ood_sample=2 \
    ood_augmentation=true generation_model=strong_augmentation \
    num_runs=1 seed=0 experiment_name=lise_smoke_train_ood_augment

python -m experiment \
    model=resnet18 ssl=simclr dataset=imagenet100_imbalanced \
    logger=false pretrain=false finetune=true \
    finetune_benchmarks='[ImageNet100LTFineTune]' \
    finetune_max_epochs=1 finetune_max_time_minutes=5 \
    limit_train_batches=1 limit_val_batches=1 limit_test_batches=1 \
    num_sanity_val_steps=0 num_runs=1 seed=0 \
    experiment_name=lise_smoke_finetune

python - <<'PY'
from PIL import Image
from experiment.ImbalancedTraining import StableDiffusion3Augmentor

augmentor = StableDiffusion3Augmentor(device="cuda")
outputs = augmentor.augment(
    [Image.new("RGB", (256, 256), (120, 80, 40))],
    num_generations_per_image=1,
    num_steps=2,
    guidance=1.0,
    strength=0.6,
    height=256,
    width=256,
)
assert len(outputs) == 1
print("LISE_SD3_SMOKE_OK", outputs[0].size)
PY

echo LISE_ALL_SMOKES_OK
