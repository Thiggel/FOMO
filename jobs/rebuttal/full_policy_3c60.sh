#!/bin/bash
#SBATCH --job-name=fomo-policy
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# 64G was not enough.  Every one of the 21 conditions died within two minutes
# of start, at the point where Stable Diffusion 3 is deserialized, and left no
# Python traceback at all, which is the signature of a cgroup kill rather than
# an exception.  SD3 holds the transformer and three text encoders in host
# memory before anything moves to the card, and the dataloader workers hold a
# copy of the dataset wrapper each.
#SBATCH --mem=110G
# Measured at 20 hours, and the partition now allows 14 days, so there is no
# reason to sit one wall-clock hour away from a kill.
#SBATCH --time=3-00:00:00
#
# This script carried no SBATCH directives and relied on every caller passing
# them, so a submission without --gres landed on a CPU-only allocation and
# every arm died deserializing the source checkpoint onto an absent device.
#
# Unified repeated-repair causal comparison from paired source checkpoints.
# Every arm receives three 60-epoch stages capped at 4,850 updates per stage.
# Repair arms add 7,500 items in total.  The one-shot arm adds the complete
# budget before its first stage while the repeated arms divide it equally.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

# Two workers leave the card idle: a survey of the running jobs found the
# 5-cycle SimCLR arms spiking to 78 percent for one sample in ten and sitting
# at zero for the rest, which is a duty cycle around eight percent.  Persistent
# workers matter as much as the count here, because a stage is a few hundred
# steps and the pool was otherwise rebuilt every epoch.
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-6}"
# Persistent workers are off again.  Keeping the pool alive across a cycle
# boundary races with the teardown of the previous cycle's combined loader:
# the run prints "terminate called without an active exception" as the stage
# ends and the worker dies of SIGABRT.  Seven runs were lost to this in one
# morning, every one of them at a cycle boundary and none inside a stage.  The
# throughput this was meant to buy comes mostly from the worker count and the
# prefetch depth below, which are unaffected.
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
# SD3 and the encoder alternate large short-lived allocations, which fragments
# the caching allocator badly enough to fail a 20 MiB request on a card with
# gigabytes free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-18000}"
# Hold the spare memory for the life of the run.  Two objective runs were lost
# to a process arriving on the card hours after the guard had checked it.
export FOMO_GPU_PEAK_MIB="${FOMO_GPU_PEAK_MIB:-18000}"
# A 24 GB card fits exactly one of these arms, so a task that lands beside a
# co-tenant has to wait for it rather than fail.  The default gives up after an
# hour, which threw away a whole allocation for a 20-hour job.  Four hours of
# waiting is cheap against that, and the wall is 48 hours.
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

conditions=(
  no_repair adaptive static one_shot uniform top_tail conventional
)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]:?Unknown full-policy task $task}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"

augment=true
selection=ood
strategy=mode_window
reuse=adaptive
repair_once=false
generations=5
generator=stable_diffusion_3

case "$condition" in
  no_repair)
    augment=false
    ;;
  adaptive)
    ;;
  static)
    reuse=static_first_cycle
    ;;
  one_shot)
    repair_once=true
    generations=15
    ;;
  uniform)
    selection=random
    ;;
  top_tail)
    strategy=top
    ;;
  conventional)
    generator=strong_augmentation
    ;;
  *)
    echo "Unknown condition $condition" >&2
    exit 2
    ;;
esac

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_fullpolicy3e60_${condition}_${seed}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=4 \
  n_epochs_per_cycle=60 max_steps_per_cycle=4850 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample="$generations" \
  sample_selection="$selection" ood_selection_strategy="$strategy" \
  selection_reuse_policy="$reuse" repair_once="$repair_once" \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model="$generator" \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
