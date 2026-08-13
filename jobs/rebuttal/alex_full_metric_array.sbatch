#!/bin/bash
#SBATCH --job-name=fomo-metrics
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/atuin/c107fa/c107fa12/FOMO_runtime/logs/full_metric_queue/alex_%A_%a.out
#
# Gruenau is the preferred cluster.  Run here only for cells whose checkpoints
# exist only on Alex, and keep the footprint small: the account is capped at
# 16 GPUs total and another project needs an 8-GPU block at a time, so submit
# at most 2-4 single-GPU tasks and never enough to starve that block.  Default
# to a40 rather than a100 for the same reason -- the 8-GPU job wants a100s.
#
# Alex caps jobs at 24 hours.  The worker therefore stops claiming new cells
# once less than FOMO_QUEUE_RESERVE_SECONDS of walltime remain, so a job is
# never killed in the middle of a benchmark it cannot finish.  Cells are
# individually idempotent, so simply resubmitting this array continues the
# queue where the previous wave stopped.

set -uo pipefail

export FOMO_CLUSTER=alex
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export FOMO_REPO_DIR="${FOMO_REPO_DIR:-$WORK/FOMO}"
export BASE_CACHE_DIR="${BASE_CACHE_DIR:-$WORK/FOMO_runtime}"
export FOMO_METRIC_QUEUE="${FOMO_METRIC_QUEUE:-$BASE_CACHE_DIR/full_metric_queue}"
export FOMO_WORKER_NAME="alex-${SLURM_ARRAY_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"

export FOMO_QUEUE_DEADLINE_SECONDS=86400
export FOMO_QUEUE_RESERVE_SECONDS=10800

# A100s are exclusive per job here, so the free-memory guard only needs to
# catch a genuinely broken card rather than a competing tenant.
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-12000}"

mkdir -p "$BASE_CACHE_DIR/logs/full_metric_queue"
bash "$FOMO_REPO_DIR/jobs/rebuttal/full_metric_queue_worker.sh"
