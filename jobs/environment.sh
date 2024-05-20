module purge
module load 2022
module load Anaconda3/2022.05

export CUBLAS_WORKSPACE_CONFIG=:4096:8

conda activate fomo

pip install -r requirements.txt
