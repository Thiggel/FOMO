. jobs/environment.sh

python -m experiment \
	--model ResNet18 \
	--imbalance_method linearly_increasing \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20
