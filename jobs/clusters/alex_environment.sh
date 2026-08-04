#!/bin/sh
set -eu

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export FOMO_REPO_DIR="${FOMO_REPO_DIR:-$WORK/FOMO}"
export BASE_CACHE_DIR="${BASE_CACHE_DIR:-$WORK/FOMO_runtime}"
export CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-$BASE_CACHE_DIR/checkpoints}"
export HF_HOME="${HF_HOME:-$BASE_CACHE_DIR/hf}"
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
export http_proxy="${http_proxy:-http://proxy:80}"
export https_proxy="${https_proxy:-http://proxy:80}"

# The shared venv is intentionally never modified. Only missing, small packages
# live in this overlay.
export PYTHONPATH="$WORK/FOMO_vendor${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$WORK/.venv/bin:$PATH"

mkdir -p "$CHECKPOINT_ROOT_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$TRANSFORMERS_CACHE" "$PYTORCH_LIGHTNING_HOME" "$WANDB_CACHE_DIR"
mkdir -p "$FOMO_DATA_ROOT" "$PYTHONPYCACHEPREFIX" "$TORCH_HOME"

if [ ! -x "$WORK/.venv/bin/python" ]; then
    echo "Missing shared Alex environment: $WORK/.venv/bin/python" >&2
    exit 1
fi
