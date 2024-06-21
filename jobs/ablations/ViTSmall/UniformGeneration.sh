. jobs/environment.sh

python -m experiment \
	--model ViTSmall \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--no-use_ood \
	--n_epochs_per_cycle 20 \
	--num_runs 1 \
	--seeds 3
