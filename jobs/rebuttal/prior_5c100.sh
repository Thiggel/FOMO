#!/bin/bash
#SBATCH --job-name=fomo-prior5c100
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-8
#
# Closest prior acquisition rules against BRIDGE on the protocol the paper
# reports: five cycles of 100 epochs, the frozen SD3 image-to-image operator,
# k=100, 500 anchors and 5 generations per anchor, one shared per-seed source
# checkpoint so that cycle-0 variance does not enter the comparison.
#
# Only the rules the paper does not already report are run.  BRIDGE is included
# as the in-batch reference, because a TADA number is only readable against a
# BRIDGE number produced by the same wave on the same encoders.
#
# The earlier comparison against TADA could not answer the question it was
# built for.  One family ran a single repair stage, which Proposition 6 says no
# acquisition rule can separate in.  The other ran six cycles but capped each
# cycle at a fixed step count, and that cap is what triggers the checkpointing
# fault: every cycle stops at the same global step, the callback skips the save
# as a duplicate, and the run publishes its first-cycle encoder.  Those runs
# score 27.18 against 27.34 for a genuine one-stage run of the same rule, which
# is how we know what they measured.  No cap is set here, so cycles end on an
# epoch boundary at different step counts and every cycle writes.
#
# Every arm shares the acquisition budget, the generation volume, the optimizer
# budget and the generator.  Only the rule that picks the anchors changes.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-6}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-18000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

conditions=(
  bridge tada cluster_inverse
)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]:?Unknown prior-baseline task $task}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"

augment=true
selection=ood
strategy=mode_window

case "$condition" in
  bridge)          ;;
  tada)            selection=early_loss ;;
  cluster_inverse) strategy=cluster_inverse ;;
  *)
    echo "Unknown condition $condition" >&2
    exit 2
    ;;
esac

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_prior5c100_${condition}_${seed}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=6 \
  n_epochs_per_cycle=100 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy="$strategy" \
  selection_reuse_policy=adaptive repair_once=false \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
