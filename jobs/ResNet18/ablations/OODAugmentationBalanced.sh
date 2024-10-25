. jobs/environment.sh

python -m experiment \
	--model ResNet18 \
	--imbalance_method no_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --crop_size 96 \
  --experiment_name "ResNet18_OODAugmentationBalanced"
