. jobs/environment.sh

python -m experiment \
	--model ViTSmall \
  --ssl_method Supervised \
	--imbalance_method no_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --crop_size 224 \
  --experiment_name "ViTSmall_OODAugmentationBalanced_Supervised"
