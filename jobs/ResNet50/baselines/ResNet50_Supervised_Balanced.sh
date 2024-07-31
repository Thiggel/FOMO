. jobs/environment.sh

python -m experiment \
  --ssl_method Supervised \
	--model_name ResNet50 \
	--imbalance_method no_imbalance \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 224 \
  --experiment_name "Baseline_ResNet50_Supervised_Balanced"
