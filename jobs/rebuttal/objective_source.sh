#!/bin/bash
#SBATCH --job-name=fomo-obj-src
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
# 24GB cards cannot hold the multi-crop batch alongside an MPS co-tenant.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=16
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

# The cluster default of 2 dataloader workers starves the GPU: multi-crop
# augmentation decodes and transforms ten views per sample on the CPU, and a
# survey of our running jobs found most GPUs at 0 percent utilisation with
# memory resident.  Persistent workers matter as much as the count, because an
# epoch here is only a few hundred steps and the pool was otherwise being torn
# down and rebuilt every epoch.
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-12}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-1}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-6}"


# Both objectives need a source here.  DINOv2 is new, and the MAE
# compatibility runs kept only result.json, so no MAE weights survive to branch
# from.  Tasks 0-2 are MAE seeds, tasks 3-5 are DINOv2 seeds.
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
if (( task < 3 )); then
  family=mae
  ssl=mae
  batch=64
  accum=2
else
  family=dinov2
  ssl=dinov2
  batch=16
  accum=8
fi

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_source_${family}_vits${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag/seed_$seed"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=vit_small ssl="$ssl" \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" \
  max_cycles=1 n_epochs_per_cycle=80 \
  train_batch_size="$batch" grad_acc_steps="$accum" val_batch_size=256 \
  ood_augmentation=false \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
