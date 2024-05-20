. jobs/environment.sh

python -m experiment --model_name ViTTiny \
  --imagenet_variant 100 \
  --ssl_method SimCLR \
  --no_augmentation \
  --early_stopping_monitor "val_acc_top5" \
  --imbalance_method power_law_imbalance \
  --batch_size 256 \
  --crop_size 224 \
  --lr 5e-4 \
  --temperature 0.7 \
  --weight_decay 1e-4 \
  --early_stopping_patience 10 \
  --n_epochs_per_cycle 100
