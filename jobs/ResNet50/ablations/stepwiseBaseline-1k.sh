. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
	--imbalance-method power_law_imbalance \
  --imagenet-variant 1k \
  --num-runs 3 \
	--max-cycles 5 \
	--ood-augmentation \
	--remove-diffusion \
	--n-epochs-per-cycle 20 \
	--pct-ood 0.15 \
  --batch-size 1024 \
  --experiment-name "ResNet50_1k_stepwiseBaseline"
