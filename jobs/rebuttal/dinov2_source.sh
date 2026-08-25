#!/bin/bash
#SBATCH --job-name=fomo-dinov2-src
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
# 24GB cards cannot hold the multi-crop batch alongside an MPS co-tenant.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#
# Common source checkpoints for the DINOv2 arm.
#
# Every other objective in the full-schedule comparison branches from a
# checkpoint that already exists, because it was trained for the compatibility
# family.  DINOv2 is new here, so its source has to be produced first: the
# paired base and BRIDGE branches must start from the *same* weights for the
# comparison to be causal, and that is only true if one run makes them.
#
# One stage, no repair.  jobs/rebuttal/full_objective_5c80.sh consumes the
# result.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_source_dinov2_vits${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag/seed_$seed"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=vit_small ssl=dinov2 \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" \
  max_cycles=1 n_epochs_per_cycle=80 \
  train_batch_size=16 grad_acc_steps=8 val_batch_size=256 \
  ood_augmentation=false \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
