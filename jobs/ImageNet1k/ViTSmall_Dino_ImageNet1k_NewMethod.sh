. jobs/environment.sh

python -m experiment \
	--model-name ViTSmall \
	--imbalance-method power_law_imbalance \
  --ssl-method Dino \
	--max-cycles 1 \
  --num-runs 1 \
  --imagenet-variant 1k \
  --pct-ood 0.15 \
  --num-cycles 15 \
	--n-epochs-per-cycle 20 \
  --batch-size 128 \
  --grad-acc-steps 32 \
  --crop-size 224 \
  --experiment-name "ViTSmall_NewMethod_Dino_ImageNet1k_Balanced"

