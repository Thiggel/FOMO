. jobs/environment.sh

python -m experiment \
  --ssl_method Supervised \
	--model_name ResNet18 \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 96 \
  --experiment_name "Baseline_ResNet18_Supervised_Imbalanced"
