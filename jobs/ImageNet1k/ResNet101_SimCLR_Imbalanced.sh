. jobs/environment.sh

python -m experiment \
	--model_name ResNet101 \
  --num_runs 3 \
  --imagenet_variant 1k \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 96 \
  --batch_size 4096 \
  --experiment_name "Baseline_ResNet101_SimCLR_Imbalanced_ImageNet1k"
