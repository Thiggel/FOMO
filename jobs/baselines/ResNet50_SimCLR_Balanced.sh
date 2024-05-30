. jobs/environment.sh

python -m experiment \
	--model_name ResNet18 \
	--imbalance_method no_imbalance \
 	--crop_size 96 \
	--max_cycles 1 \
	--n_epochs_per_cycle 100
