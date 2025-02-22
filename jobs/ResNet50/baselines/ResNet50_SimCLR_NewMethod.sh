. jobs/environment.sh

srun python -m experiment \
	--model-name ResNet50 \
	--imbalance-method power_law_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 100 \
  --train-batch-size 1024 \
  --val-batch-size 1024 \
	--ood-augmentation \
	--max-cycles 5 \
	--n-epochs-per-cycle 20 \
  --num-ood-samples 500 \
  --num-generations-per-sample 5 \
  --experiment-name "Baseline_ResNet50_SimCLR_NewMethod"
