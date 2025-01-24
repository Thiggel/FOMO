. jobs/environment.sh

python -m experiment \
	--model-name ResNet50 \
  --num-runs 3 \
  --imagenet-variant 1k \
	--imbalance-method no_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 100 \
  --batch-size 4096 \
  --experiment-name "Baseline_ResNet50_SimCLR_Balanced_ImageNet1k"

