. jobs/environment.sh

python -m experiment \
<<<<<<< HEAD:jobs/ViTBase/ablations/OODAugmentationBalanced.sh
	--model ViTBase \
	--imbalance_method no_imbalance \
=======
	--model ResNet18 \
	--imbalance_method power_law_imbalance \
>>>>>>> f6dbd7cfbc211eea89b4c0701e0db2c335087910:jobs/baselines/OODAugmentationImbalanced.sh
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --crop_size 224 \
  --experiment_name "ViTBase_OODAugmentationBalanced"
