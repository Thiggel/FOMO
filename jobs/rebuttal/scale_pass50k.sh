#!/bin/bash
#SBATCH --job-name=fomo-pass50k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-5

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

variants=(base bridge)
seed="$((SLURM_ARRAY_TASK_ID % 3))"
variant="${variants[$((SLURM_ARRAY_TASK_ID / 3))]}"
augment=false
cycles=1
steps=9700
if [[ "$variant" == bridge ]]; then augment=true; cycles=2; steps=4850; fi
run_root="$BASE_CACHE_DIR/rebuttal_runs/pass50k/$variant/seed_$seed"
mkdir -p "$run_root"
python -m experiment \
    dataset=pass_subset dataset.split='train[:50000]' model=resnet50 ssl=simclr logger=false \
    pretrain=true finetune=true finetune_benchmarks='[CarsFineTune,AircraftFineTune,ImageNet100LTFineTune]' \
    num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
    train_batch_size=128 val_batch_size=256 ood_augmentation="$augment" generation_model=strong_augmentation \
    ood_distance_metric=normalized_l2 additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_pass50k_${variant}"
