. jobs/environment.sh

python -m experiment \
	--model ViTBase \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--no-use_ood \
	--n_epochs_per_cycle 20 \
	--num_runs 1 \
	--seeds 3 \
  --crop_size 224 \
  --experiment_name "ViTBase_UniformGeneration"
