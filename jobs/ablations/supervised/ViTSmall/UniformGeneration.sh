. jobs/environment.sh

python -m experiment \
	--model ViTSmall \
  --ssl_method Supervised \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--no-use_ood \
	--n_epochs_per_cycle 20 \
  --crop_size 224 \
  --experiment_name "ViTSmall_UniformGeneration_Supervised"
