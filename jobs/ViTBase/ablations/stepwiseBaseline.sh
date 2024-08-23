. jobs/environment.sh

python -m experiment \
<<<<<<< HEAD:jobs/ViTBase/ablations/stepwiseBaseline.sh
	--model ViTBase \
=======
	--model ResNet50 \
>>>>>>> f6dbd7cfbc211eea89b4c0701e0db2c335087910:jobs/ablations/ResNet50/stepwiseBaseline.sh
	--imbalance_method power_law_imbalance \
	--max_cycles 5 \
	--ood_augmentation \
	--remove_diffusion \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
  --crop_size 224 \
  --experiment_name "ViTBase_stepwiseBaseline"
