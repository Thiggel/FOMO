. jobs/environment.sh

srun python -m experiment \
	--model ViTSmall \
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
       --pct_ood 0.15 \
       --crop_size 224 \
  --experiment_name "NewMethod_ViTSmall_SimCLR_Imbalanced"
