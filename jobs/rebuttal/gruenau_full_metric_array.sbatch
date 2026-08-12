#!/bin/bash
#SBATCH --job-name=fomo-metrics
#SBATCH --partition=wbimlgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=guppi1,guppi4
#SBATCH --output=/vol/home-vol2/ml/laitenbf/FOMO_runtime/logs/full_metric_queue/gruenau_%A_%a.out
#
# One worker per allocated GPU.  Slurm itself is the "find a free GPU" loop:
# extra array tasks simply pend until a GPU frees up, and each task pins
# exactly one device, which is the configuration the benchmark stage requires.
#
# Site policy rewrites GPU jobs onto wbimlgpu (guppi[1,4-8]), so the partition
# is named explicitly.  guppi1 and guppi4 are excluded because their TaskProlog
# fails with status 127 and silently kills every task placed on them.

set -uo pipefail

export FOMO_CLUSTER=gruenau
export FOMO_REPO_DIR="${FOMO_REPO_DIR:-/vol/home-vol2/ml/laitenbf/FOMO}"
export BASE_CACHE_DIR="${BASE_CACHE_DIR:-/vol/home-vol2/ml/laitenbf/FOMO_runtime}"
export FOMO_METRIC_QUEUE="${FOMO_METRIC_QUEUE:-$BASE_CACHE_DIR/full_metric_queue}"
export FOMO_WORKER_NAME="gruenau-${SLURM_ARRAY_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"

# Gruenau jobs are not walltime-bound in practice, but keep a reserve so a
# worker never dies in the middle of a benchmark it could not finish.
export FOMO_QUEUE_DEADLINE_SECONDS=345600
export FOMO_QUEUE_RESERVE_SECONDS=9000

# The smallest cards in the gpu partition are 24 GB; require nearly all of it
# so a worker never lands on a card another user is already occupying.
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-20000}"

mkdir -p "$BASE_CACHE_DIR/logs/full_metric_queue"
bash "$FOMO_REPO_DIR/jobs/rebuttal/full_metric_queue_worker.sh"
