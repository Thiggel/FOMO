. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
  --ssl_method Supervised \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--no-use_ood \
	--n_epochs_per_cycle 20 \
  --experiment_name "ResNet50_UniformGeneration_Supervised"
