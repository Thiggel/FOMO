. jobs/environment.sh

python -m experiment \
	--model_name ResNet50 \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 96 \
  --experiment_name "Baseline_ResNet50_SimCLR_Imbalanced"