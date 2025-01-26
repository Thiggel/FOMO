. jobs/environment.sh

python -m experiment \
	--model-name ResNet50 \
  --num-runs 3 \
  --imagenet-variant 1k \
	--imbalance-method power_law_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 100 \
  --batch-size 1024 \
  --experiment_name "Baseline_ResNet50_SimCLR_Imbalanced_ImageNet1k"
