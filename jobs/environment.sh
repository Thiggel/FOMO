module purge
module load gcc/default
if ! module load cuda/12.1/12.1.1 2>/dev/null; then
    module load cuda/12.8.1
fi

export http_proxy="http://proxy:80"
export https_proxy="http://proxy:80"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export BASE_CACHE_DIR="$WORK/FOMO2"
export CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-$BASE_CACHE_DIR/checkpoints}"
export VENV_DIR="$BASE_CACHE_DIR/.venv"
export UV_CACHE_DIR="$BASE_CACHE_DIR/uv-cache"
export PIP_CACHE_DIR="$BASE_CACHE_DIR/pip-cache"
export XDG_CACHE_HOME="$BASE_CACHE_DIR/xdg-cache"
export XDG_CONFIG_HOME="$BASE_CACHE_DIR/xdg-config"
export XDG_DATA_HOME="$BASE_CACHE_DIR/xdg-data"
export PYTHONPYCACHEPREFIX="$BASE_CACHE_DIR/python-pycache"
export MPLCONFIGDIR="$BASE_CACHE_DIR/matplotlib"
export TORCH_HOME="$BASE_CACHE_DIR/torch"
export TORCH_EXTENSIONS_DIR="$BASE_CACHE_DIR/torch-extensions"
export CUDA_CACHE_PATH="$BASE_CACHE_DIR/nv/ComputeCache"
export NVIDIA_CACHE_DIR="$BASE_CACHE_DIR/nv"
: "${TMPDIR:?TMPDIR is not set. Expected scheduler-provided scratch directory.}"
export TMPDIR

mkdir -p "$BASE_CACHE_DIR"
mkdir -p "$CHECKPOINT_ROOT_DIR"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$PYTHONPYCACHEPREFIX"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$TMPDIR"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$NVIDIA_CACHE_DIR"
mkdir -p "$BASE_CACHE_DIR/hf" "$BASE_CACHE_DIR/lightning_logs"

export FOMO_DATASET_TMPDIR="$TMPDIR/fomo2-datasets"
mkdir -p "$FOMO_DATASET_TMPDIR"
export FOMO_HF_CACHE_ROOT="$FOMO_DATASET_TMPDIR/hf-cache"
export FOMO_HF_DATASETS_CACHE="$FOMO_HF_CACHE_ROOT/datasets"
export FOMO_HF_HUB_CACHE="$FOMO_HF_CACHE_ROOT/hub"
export FOMO_HF_MODULES_CACHE="$FOMO_HF_CACHE_ROOT/modules"
export FOMO_TRANSFORMERS_CACHE="$FOMO_HF_CACHE_ROOT/transformers"
export FOMO_HF_IMAGE_CACHE="$FOMO_HF_CACHE_ROOT/image-cache"
mkdir -p "$FOMO_HF_DATASETS_CACHE" "$FOMO_HF_HUB_CACHE" "$FOMO_HF_MODULES_CACHE"
mkdir -p "$FOMO_TRANSFORMERS_CACHE" "$FOMO_HF_IMAGE_CACHE"
export FOMO_ALLOW_DATA_DOWNLOAD="${FOMO_ALLOW_DATA_DOWNLOAD:-0}"

if [ -f "$BASE_CACHE_DIR/data.tar" ]; then
    if [ ! -d "$FOMO_DATASET_TMPDIR/data" ]; then
        tar -xf "$BASE_CACHE_DIR/data.tar" -C "$FOMO_DATASET_TMPDIR"
    fi
fi

if [ -d "$FOMO_DATASET_TMPDIR/data" ]; then
    export FOMO_DATA_ROOT="$FOMO_DATASET_TMPDIR/data"
elif [ -d "$FOMO_DATASET_TMPDIR/cifar-10-batches-py" ] || [ -d "$FOMO_DATASET_TMPDIR/fgvc-aircraft-2013b" ]; then
    export FOMO_DATA_ROOT="$FOMO_DATASET_TMPDIR"
fi

if [ -f "$BASE_CACHE_DIR/stanford_cars.tar" ]; then
    if [ ! -d "$FOMO_DATASET_TMPDIR/stanford_cars" ] && [ ! -d "$FOMO_DATASET_TMPDIR/cars_train" ]; then
        tar -xf "$BASE_CACHE_DIR/stanford_cars.tar" -C "$FOMO_DATASET_TMPDIR"
    fi

    if [ -d "$FOMO_DATASET_TMPDIR/stanford_cars" ]; then
        export STANFORD_CARS_ROOT="$FOMO_DATASET_TMPDIR/stanford_cars"
    elif [ -d "$FOMO_DATASET_TMPDIR/cars_train" ] && [ -d "$FOMO_DATASET_TMPDIR/devkit" ]; then
        export STANFORD_CARS_ROOT="$FOMO_DATASET_TMPDIR"
    else
        echo "Stanford Cars extraction failed under $FOMO_DATASET_TMPDIR" >&2
        return 1 2>/dev/null || exit 1
    fi
fi

if [ -n "${FOMO_DATA_ROOT:-}" ]; then
    echo "FOMO_DATA_ROOT=$FOMO_DATA_ROOT"
fi
if [ -n "${STANFORD_CARS_ROOT:-}" ]; then
    echo "STANFORD_CARS_ROOT=$STANFORD_CARS_ROOT"
fi

export HF_HOME="$BASE_CACHE_DIR/hf"
export HF_HUB_CACHE="$FOMO_HF_HUB_CACHE"
export HF_DATASETS_CACHE="$FOMO_HF_DATASETS_CACHE"
export TRANSFORMERS_CACHE="$FOMO_TRANSFORMERS_CACHE"
export HF_MODULES_CACHE="$FOMO_HF_MODULES_CACHE"
export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb/cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb/config"
export WANDB_DATA_DIR="$TMPDIR/wandb/data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

. "$VENV_DIR/bin/activate"
