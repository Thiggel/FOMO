. jobs/environment.sh

python -m experiment \
	--model_name ResNet50 \
  --num_runs 1 \
  --imagenet_variant 1k \
	--imbalance_method no_imbalance \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 96 \
  --experiment_name "Baseline_ResNet50_SimCLR_Balanced_ImageNet1k"

