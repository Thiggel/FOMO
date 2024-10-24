. jobs/environment.sh

python -m experiment \
	--model ViTBase \
	--imbalance_method power_law_imbalance \
  --ssl_method Dino \
	--max_cycles 5 \
	--ood_augmentation \
	--n_epochs_per_cycle 20 \
	--pct_ood 0.15 \
	--crop_size 224 \
  --num_runs 1 \
  --imagenet_variant 1k \
  --experiment_name "NewMethod_ViTBase_SimCLR_Imbalanced_ImageNet1k"
