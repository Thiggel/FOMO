. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--imbalance_method power_law_imbalance \
  --ssl_method Dino \
	--max_cycles 1 \
  --num_runs 1 \
  --imagenet_variant 1k \
  --pct-ood 0.15 \
  --num-cycles 5 \
	--n_epochs_per_cycle 20 \
  --batch_size 32 \
  --grad_acc_steps 32 \
  --crop_size 224 \
  --experiment_name "ViTSmall_NewMethod_Dino_ImageNet1k_Balanced"

