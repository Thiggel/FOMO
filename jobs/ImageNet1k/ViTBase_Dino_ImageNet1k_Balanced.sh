. jobs/environment.sh

python -m experiment \
	--model-name ViTBase \
	--imbalance-method no_imbalance \
  --ssl-method Dino \
	--max-cycles 1 \
  --num-runs 3 \
  --imagenet-variant 1k \
	--n-epochs-per-cycle 300 \
  --batch-size 1024 \
  --grad-acc-steps 1 \
  --crop-size 224 \
  --experiment-name "Baseline_ViTBase_Dino_ImageNet1k_Balanced"

. jobs/environment.sh

