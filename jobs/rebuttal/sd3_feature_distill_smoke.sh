#!/bin/bash
#SBATCH --job-name=fomo-sd3-distill-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
cache="$BASE_CACHE_DIR/sd3_teacher_cache/seed_0.pt"; test -f "$cache"
root="$BASE_CACHE_DIR/rebuttal_runs/smokes/sd3_vae_distill"; mkdir -p "$root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=false \
  num_runs=1 seed=0 max_cycles=1 n_epochs_per_cycle=1 max_steps_per_cycle=1 \
  train_batch_size=8 val_batch_size=8 limit_train_batches=1 limit_val_batches=0 num_sanity_val_steps=0 \
  external_diffusion_teacher=true external_diffusion_teacher_cache="$cache" external_teacher_weight=0.1 \
  additional_data_path="$root/generated" experiment_name=rebuttal_smoke_sd3_vae_distill
