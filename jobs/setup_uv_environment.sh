#!/bin/sh
set -eu

REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BASE_CACHE_DIR="$WORK/FOMO2"
VENV_DIR="$BASE_CACHE_DIR/.venv"

mkdir -p "$REPO_DIR/job_logs"
mkdir -p "$BASE_CACHE_DIR"

export UV_CACHE_DIR="$BASE_CACHE_DIR/uv-cache"
export UV_NO_CACHE=1
export PIP_CACHE_DIR="$BASE_CACHE_DIR/pip-cache"
export PIP_NO_CACHE_DIR=1
export XDG_CACHE_HOME="$BASE_CACHE_DIR/xdg-cache"
export XDG_CONFIG_HOME="$BASE_CACHE_DIR/xdg-config"
export XDG_DATA_HOME="$BASE_CACHE_DIR/xdg-data"
export PYTHONPYCACHEPREFIX="$BASE_CACHE_DIR/python-pycache"
export CUDA_CACHE_PATH="$BASE_CACHE_DIR/nv/ComputeCache"
export NVIDIA_CACHE_DIR="$BASE_CACHE_DIR/nv"
: "${TMPDIR:?TMPDIR is not set. Expected scheduler-provided scratch directory.}"
export TMPDIR

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
mkdir -p "$PYTHONPYCACHEPREFIX" "$CUDA_CACHE_PATH" "$NVIDIA_CACHE_DIR" "$TMPDIR"

cd "$REPO_DIR"

if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Using existing virtual environment at $VENV_DIR"
elif [ -x "$WORK/.venv/bin/python" ]; then
    echo "Creating virtual environment from existing interpreter at $WORK/.venv/bin/python"
    uv venv --python "$WORK/.venv/bin/python" "$VENV_DIR"
else
    uv venv --python 3.10 "$VENV_DIR"
fi

uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
