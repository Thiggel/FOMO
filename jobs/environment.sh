
conda init

source ~/.bashrc

conda activate fomo


ulimit -n 4096

export SCRATCH_LOCAL="/var/scratch/ssalehid/"

cd $SCRATCH_LOCAL
mkdir FOMO

cd $HOME/FOMO

# Base directory
export BASE_CACHE_DIR="$SCRATCH_LOCAL/FOMO"

export TMPDIR="$BASE_CACHE_DIR/tmp"

# Hugging Face
export HF_HOME="$BASE_CACHE_DIR"
export HF_DATASETS_CACHE="$BASE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$BASE_CACHE_DIR/transformers"
export HF_MODULES_CACHE="$BASE_CACHE_DIR/modules"

# DeepSpeed
export DEEPSPEED_CACHE_DIR="$BASE_CACHE_DIR/deepspeed"

# Weights & Biases
export WANDB_DIR="$BASE_CACHE_DIR/wandb"

# PyTorch Lightning
# Note: PyTorch Lightning doesn't use an environment variable, 
# but you can use this in your Python code
export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

conda activate fomo

pip install -r requirements.txt
