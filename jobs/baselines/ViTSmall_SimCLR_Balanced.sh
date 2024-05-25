. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--imbalance_method no_imbalance \
	--max_cycles 1 \
	--n_epochs_per_cycle 100
	--num_runs 2 \
	--seeds 1 2