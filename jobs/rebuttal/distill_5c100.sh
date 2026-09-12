#!/bin/bash
#SBATCH --job-name=fomo-distill5c100
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5%4
#
# Is the benefit distillation from the frozen generator, or the placement of
# the repair budget?  Both arms here draw on the same SD3 checkpoint.  One
# receives its knowledge directly, as a VAE feature-regression target on every
# training image, and the other only through images generated at the anchors
# \method selects.  If direct transfer matched targeted repair, the loop would
# be an expensive way to distil.
#
# The existing distillation control ran one or two cycles with a per-cycle step
# cap, which is the configuration that triggers the checkpointing fault, so it
# published a first-cycle encoder and could not be compared with anything.  No
# cap is set here and the schedule is the one the paper reports.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-2}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-14000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
if (( task < 3 )); then condition=distill_only; augment=false; else condition=distill_bridge; augment=true; fi

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
cache="$BASE_CACHE_DIR/sd3_teacher_cache/seed_${seed}.pt"
test -f "$cache"

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_distill5c100_${condition}_${seed}${run_suffix}"
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
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive repair_once=false \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model=stable_diffusion_3 \
  external_diffusion_teacher=true \
  external_diffusion_teacher_cache="$cache" \
  external_teacher_weight=0.1 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
