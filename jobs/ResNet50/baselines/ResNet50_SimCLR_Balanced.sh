. jobs/environment.sh

srun python -m experiment \
	--model-name ResNet50 \
	--imbalance-method no_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 100 \
  --train-batch-size 1024 \
  --val-batch-size 1024 \
  --experiment-name "Baseline_ResNet50_SimCLR_Balanced"

