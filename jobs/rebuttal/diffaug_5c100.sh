#!/bin/bash
#SBATCH --job-name=fomo-diffaug5c100
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
# DiffAug carries a learned generator beside the encoder, so it does not fit a
# 24 GiB card beside a co-tenant: seed 1 of the repair arm reached 10.5 GiB and
# died next to a neighbour holding 11.5 GiB of a 21.98 GiB card.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#
# DiffAug, the closest iterative SSL antecedent, on the protocol the paper reports,
# five cycles of 100 epochs.  DiffAug changes the SSL objective, learning a generator
# for positive views, so it cannot be an arm of the acquisition sweep in
# prior_5c80.sh, which varies the anchor rule while holding the objective
# fixed.  It gets its own paired comparison instead: DiffAug alone against
# DiffAug with \method repair, from one shared source checkpoint per seed.
#
# The earlier DiffAug runs capped each cycle at a fixed step count, which is
# what triggers the checkpointing fault, so they published first-cycle
# encoders.  No cap is set here.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-2}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-22000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
if (( task < 3 )); then condition=base; augment=false; else condition=bridge; augment=true; fi

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_diffaug5c100_${condition}_${seed}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=diffaug \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=6 \
  n_epochs_per_cycle=100 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive repair_once=false \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
