module purge
module load gcc/default
module load cuda/12.1/12.1.1

export SCRATCH_LOCAL="/groups/gag51492/users"

cd $SCRATCH_LOCAL
mkdir FOMO2

cd $HOME/FOMO

# Base directory
export BASE_CACHE_DIR="$SCRATCH_LOCAL/FOMO2"

# Hugging Face
export HF_HOME="$BASE_CACHE_DIR"
export HF_DATASETS_CACHE="$BASE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$BASE_CACHE_DIR/transformers"
export HF_MODULES_CACHE="$BASE_CACHE_DIR/modules"
export TMPDIR="/tmp"

# Weights & Biases
export WANDB_DIR="$BASE_CACHE_DIR"

# PyTorch Lightning
# Note: PyTorch Lightning doesn't use an environment variable, 
# but you can use this in your Python code
export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

source .venv/bin/activate

python -m pip install -r requirements.txt
