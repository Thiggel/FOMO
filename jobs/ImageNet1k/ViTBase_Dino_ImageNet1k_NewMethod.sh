. jobs/environment.sh

python -m experiment \
	--model_name ViTBase \
  --ssl_method Dino \
  --num_runs 3 \
  --imagenet_variant 1k \
  --max_cycles 15 \
	--n_epochs_per_cycle 20 \
  --pct_ood 0.15 \
  --batch_size 256 \
  --grad_acc_steps 4 \
  --crop_size 224 \
  --experiment_name "Baseline_ViTBase_Dino_ImageNet1k_NewMethod"
