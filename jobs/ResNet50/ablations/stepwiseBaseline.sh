. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--remove_diffusion \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --crop_size 96 \
  --experiment_name "ResNet50_stepwiseBaseline"
