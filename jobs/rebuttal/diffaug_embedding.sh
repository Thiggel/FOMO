#!/bin/bash
#SBATCH --job-name=fomo-diffaug-embed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
# Explicit image-SSL adaptation of DiffAug's encoder-conditioned diffusion
# positive mechanism; the available upstream release is non-image modality.
seed="$((SLURM_ARRAY_TASK_ID % 3))"; bridge=false; [[ "$SLURM_ARRAY_TASK_ID" -ge 3 ]] && bridge=true
cycles=1; steps=9700; [[ "$bridge" == true ]] && { cycles=2; steps=4850; }
root="$BASE_CACHE_DIR/rebuttal_runs/diffaug_embedding/$bridge/seed_$seed"; mkdir -p "$root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=diffaug logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
  train_batch_size=128 val_batch_size=256 ood_augmentation="$bridge" generation_model=stable_diffusion_3 \
  ood_distance_metric=normalized_l2 additional_data_path="$root/generated" experiment_name="rebuttal_diffaug_embedding_$bridge"
