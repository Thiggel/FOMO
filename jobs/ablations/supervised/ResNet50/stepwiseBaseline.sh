. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
  --ssl_method Supervised \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--remove_diffusion \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --experiment_name "ResNet50_stepwiseBaseline_Supervised"
