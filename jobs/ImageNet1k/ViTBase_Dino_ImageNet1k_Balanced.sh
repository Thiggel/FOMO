. jobs/environment.sh

python -m experiment \
	--model_name ViTBase \
	--imbalance_method no_imbalance \
  --ssl_method Dino \
	--max_cycles 1 \
  --num_runs 2 \
  --seeds 1, 2 \
  --imagenet_variant 1k \
	--n_epochs_per_cycle 300 \
  --batch_size 32 \
  --grad_acc_steps 32 \
  --crop_size 224 \
  --experiment_name "Baseline_ViTBase_Dino_ImageNet1k_Balanced"

