. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 224 \
  --experiment_name "Baseline_ViTSmall_SimCLR_Imbalanced"
