. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--imbalance_method no_imbalance \
  --ssl_method Dino \
	--max_cycles 1 \
  --num_runs 1 \
  --imagenet_variant 1k \
	--n_epochs_per_cycle 300 \
  --crop_size 224 \
  --batch_size 64 \
  --grad_acc_steps 16 \
  --experiment_name "Baseline_ViTBase_Dino_ImageNet1k_Balanced"

