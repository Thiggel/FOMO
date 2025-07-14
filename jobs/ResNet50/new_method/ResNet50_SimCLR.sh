. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
       --pct_ood 0.15 \
       --crop_size 96 \
  --experiment_name "NewMethod_ResNet50_SimCLR_Imbalanced"
