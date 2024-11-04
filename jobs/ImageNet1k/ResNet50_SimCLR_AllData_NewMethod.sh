. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
  --num_runs 1 \
  --imagenet_variant 1k \
	--imbalance_method all_data \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.075 \
	--crop_size 96 \
  --experiment_name "NewMethod_ResNet50_SimCLR_Imbalanced_ImagNet1k_AllData"
