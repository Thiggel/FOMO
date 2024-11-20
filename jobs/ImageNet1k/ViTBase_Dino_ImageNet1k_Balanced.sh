. jobs/environment.sh

python -m experiment \
	--model_name ViTBase \
	--imbalance_method no_imbalance \
  --ssl_method Dino \
	--max_cycles 1 \
  --num_runs 1 \
  --imagenet_variant 1k \
	--n_epochs_per_cycle 300 \
  --batch_size 64 \
  --grad_acc_steps 16 \
  --crop_size 224 \
  --experiment_name "Baseline_ViTBase_Dino_ImageNet1k_Balanced"

