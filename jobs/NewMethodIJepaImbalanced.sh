. jobs/environment.sh

python -m experiment --model_name ViTTinyJeppa \
  --imagenet_variant 100 \
  --ssl_method "I-Jepa" \
  --no_augmentation \
  --early_stopping_monitor "val_loss" \
  --imbalance_method power_law_imbalance \
  --batch_size 4 \
  --crop_size 224 \
  --lr 1e-3 \
  --temperature 0.7 \
  --weight_decay 0.04 \
  --early_stopping_patience 10 \
  --pretrain \
  --no-finetune \
  --n_epochs_per_cycle 5 \
  --max_cycles 20 \
  --logger
