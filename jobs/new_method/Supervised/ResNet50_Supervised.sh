. jobs/environment.sh

python -m experiment \
  --ssl_method Supervised \
	--model ResNet50 \
	--imbalance_method linearly_increasing \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
	--crop_size 96
