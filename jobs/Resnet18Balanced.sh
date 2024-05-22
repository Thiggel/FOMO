. jobs/environment.sh

python -m experiment --model_name ResNet18 \
  --imagenet_variant 100 \
  --ssl_method SimCLR \
  --no_augmentation \
  --classification_head \
  --early_stopping_monitor "val_acc_top5" \
  --imbalance_method no_imbalance \
  --batch_size 256 \
  --crop_size 96 \
  --lr 5e-4 \
  --temperature 0.7 \
  --weight_decay 1e-4 \
  --early_stopping_patience 10 \
  --pretrain \
  --finetune \
  --n_epochs_per_cycle 100 \
  --max_cycles 1 \
  --logger
