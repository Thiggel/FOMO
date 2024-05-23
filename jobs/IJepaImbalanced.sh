. jobs/environment.sh

python -m experiment --model_name ViTTinyJeppa \
  --imagenet_variant 100 \
  --ssl_method "I-Jepa" \
  --no_augmentation \
  --warmup 700 \
  --early_stopping_monitor "val_loss" \
  --imbalance_method linearly_increasing \
  --batch_size 128 \
  --crop_size 224 \
  --lr 3e-3 \
  --final_lr 1e-4 \
  --temperature 0.7 \
  --weight_decay 0.04 \
  --n_epochs_per_cycle 100 \
  --max_cycles 1 \
  --num_runs 3
