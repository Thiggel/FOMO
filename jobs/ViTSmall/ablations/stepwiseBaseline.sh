. jobs/environment.sh

python -m experiment \
	--model ViTSmall \
	--imbalance-method power_law_imbalance \
	--max-cycles 5 \
	--ood-augmentation \
	--remove-diffusion \
	--n-epochs-per-cycle 20 \
	--pct-ood 0.15 \
  --crop-size 224 \
  --experiment-name "ViTSmall_stepwiseBaseline"
