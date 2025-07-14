. jobs/environment.sh

python -m experiment \
        --model ResNet50 \
        --imbalance_method power_law_imbalance \
        --max_cycles 5 \
        --ood_augmentation \
        --n_epochs_per_cycle 20 \
        --pct_ood 0.15 \
        --crop_size 96 \
        --use-temperature-schedule \
        --temperature-min 0.1 \
        --temperature-max 1.0 \
        --t-max 400 \
  --experiment_name "NewMethod_ResNet50_SimCLR_Bridge_Temperature"
