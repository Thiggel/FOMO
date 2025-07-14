. jobs/environment.sh

python -m experiment \
	--model_name ResNet101 \
  --num_runs 3 \
  --imagenet_variant 1k \
	--max_cycles 5 \
	--ood_augmentation \
        --n_epochs_per_cycle 20 \
        --pct_ood 0.15 \
  --crop_size 96 \
  --batch_size 4096 \
        --use-temperature-schedule \
        --temperature-min 0.1 \
        --temperature-max 1.0 \
        --t-max 400 \
  --experiment_name "Baseline_ResNet101_SimCLR_NewMethod_ImageNet1k"
