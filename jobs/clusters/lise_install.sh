#!/bin/bash
#SBATCH --job-name=fomo-install
#SBATCH --partition=cpu-clx
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/usr/bemflait/FOMO_runtime/logs/install-%j.out

set -euo pipefail
: "${WORK:?WORK must be set on Lise}"

REPO_DIR="$WORK/FOMO"
RUNTIME_DIR="$WORK/FOMO_runtime"
VENV_DIR="$RUNTIME_DIR/.venv"
export UV_CACHE_DIR="$RUNTIME_DIR/uv-cache"
export PATH="$WORK/bin:$PATH"
mkdir -p "$RUNTIME_DIR/logs" "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    uv venv --python "$WORK/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10" "$VENV_DIR"
fi

uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt"
uv pip install --python "$VENV_DIR/bin/python" --reinstall \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126

"$VENV_DIR/bin/python" -m compileall -q "$REPO_DIR/experiment"
"$VENV_DIR/bin/python" - <<'PY'
import torch
import torchvision
import lightning
import datasets
import diffusers
import faiss
import timm
print(
    "LISE_INSTALL_OK",
    torch.__version__,
    torchvision.__version__,
    lightning.__version__,
)
PY
