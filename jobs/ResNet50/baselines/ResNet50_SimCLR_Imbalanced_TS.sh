. jobs/environment.sh

srun python -m experiment \
	--model-name ResNet50 \
	--imbalance-method power_law_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 800 \
  --train-batch-size 512 \
  --val-batch-size 512 \
  --use-temperature-schedule \
  --experiment-name "Baseline_ResNet50_SimCLR_Imbalanced_TS"
