#!/bin/bash
#SBATCH --job-name=fomo-ts-interaction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(base ts bridge bridge_ts)
task="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
use_ts=false
augment=false
cycles=1
steps=9700
if [[ "$condition" == ts || "$condition" == bridge_ts ]]; then use_ts=true; fi
if [[ "$condition" == bridge || "$condition" == bridge_ts ]]; then augment=true; cycles=2; steps=4850; fi
run_root="$BASE_CACHE_DIR/rebuttal_runs/ts_interaction/$condition/seed_${seed}${run_suffix}"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
    train_batch_size=128 val_batch_size=256 use_temperature_schedule="$use_ts" \
    ood_augmentation="$augment" generation_model=stable_diffusion_3 ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" experiment_name="rebuttal_ts_${condition}${run_suffix}"
