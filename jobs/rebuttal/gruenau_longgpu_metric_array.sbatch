#!/bin/bash
#SBATCH --job-name=fomo-metrics
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=gruenau7
#SBATCH --output=/vol/home-vol2/ml/laitenbf/FOMO_runtime/logs/full_metric_queue/longgpu_%A_%a.out
#
# Second pool of workers on the same shared queue.
#
# The site submit plugin rewrites --partition=gpu to wbimlgpu, which restricts
# jobs to the guppi nodes.  It does not rewrite longgpu, which reaches
# gruenau[1-2,9-10] and their V100, RTX6000 and A100 cards.  Those are a
# meaningful throughput addition and, in the A100 case, considerably faster
# than the RTX3090s.
#
# Workers claim from the queue atomically, so this pool and the wbimlgpu pool
# can run concurrently without coordinating.
#
# gruenau7 is excluded: every cell placed there died with a DataLoader worker
# timeout during feature extraction (12 failures, 0 completions across four
# workers), while the same cells succeed elsewhere.  It is not a memory or
# shared-memory limit -- /dev/shm was 1% used and the node has 1 TB of RAM --
# so it looks like a sick node rather than a resource ceiling.  The node had
# also been in a "down" Slurm state earlier in the day.

set -uo pipefail

export FOMO_CLUSTER=gruenau
export FOMO_REPO_DIR="${FOMO_REPO_DIR:-/vol/home-vol2/ml/laitenbf/FOMO}"
export BASE_CACHE_DIR="${BASE_CACHE_DIR:-/vol/home-vol2/ml/laitenbf/FOMO_runtime}"
export FOMO_METRIC_QUEUE="${FOMO_METRIC_QUEUE:-$BASE_CACHE_DIR/full_metric_queue}"
export FOMO_WORKER_NAME="longgpu-${SLURM_ARRAY_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"

export FOMO_QUEUE_DEADLINE_SECONDS=345600
export FOMO_QUEUE_RESERVE_SECONDS=9000

# These nodes are shared with other tenants, including non-Slurm processes, so
# keep the tolerant threshold and wait rather than abandoning an allocation.
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-12000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-60}"

mkdir -p "$BASE_CACHE_DIR/logs/full_metric_queue"
bash "$FOMO_REPO_DIR/jobs/rebuttal/full_metric_queue_worker.sh"
