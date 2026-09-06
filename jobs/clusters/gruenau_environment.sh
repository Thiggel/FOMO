#!/bin/sh
set -eu

REPO_DIR="${FOMO_REPO_DIR:-/vol/home-vol2/ml/laitenbf/FOMO}"
export BASE_CACHE_DIR="${BASE_CACHE_DIR:-/vol/home-vol2/ml/laitenbf/FOMO_runtime}"
export CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-$BASE_CACHE_DIR/checkpoints}"
# Set, not defaulted.  sbatch propagates the submitting shell, and ~/.bashrc
# exports HF_HOME=/vol/tmp/laitenbf for interactive work, so every job so far
# has been reading its weights and datasets from /vol/tmp.  That volume is
# being retired and its contents are not guaranteed to survive the upgrade.
# The caches a run needs all live under BASE_CACHE_DIR, so point there
# regardless of what the submitting environment carried in.
export HF_HOME="$BASE_CACHE_DIR/hf"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTORCH_LIGHTNING_HOME="${PYTORCH_LIGHTNING_HOME:-$BASE_CACHE_DIR/lightning}"
export WANDB_DIR="${WANDB_DIR:-$BASE_CACHE_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export FOMO_DATA_ROOT="${FOMO_DATA_ROOT:-$BASE_CACHE_DIR/data}"
export STANFORD_CARS_ROOT="${STANFORD_CARS_ROOT:-$BASE_CACHE_DIR/stanford_cars}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$BASE_CACHE_DIR/pycache}"
export TORCH_HOME="${TORCH_HOME:-$BASE_CACHE_DIR/torch}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-2}"
export FOMO_START_METHOD="${FOMO_START_METHOD:-fork}"

mkdir -p "$CHECKPOINT_ROOT_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$TRANSFORMERS_CACHE" "$PYTORCH_LIGHTNING_HOME" "$WANDB_CACHE_DIR"
mkdir -p "$FOMO_DATA_ROOT" "$PYTHONPYCACHEPREFIX" "$TORCH_HOME"

if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
    echo "Missing gruenau environment: $REPO_DIR/.venv/bin/python" >&2
    exit 1
fi

export PATH="$REPO_DIR/.venv/bin:$PATH"
