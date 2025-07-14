. jobs/environment.sh

srun python -m experiment \
        --model-name ResNet50 \
        --imbalance-method power_law_imbalance \
        --max-cycles 1 \
        --n-epochs-per-cycle 100 \
        --use-temperature-schedule \
        --temperature-min 0.1 \
        --temperature-max 1.0 \
        --t-max 400 \
  --train-batch-size 1024 \
  --val-batch-size 1024 \
  --experiment-name "Baseline_ResNet50_SimCLR_TemperatureSchedule"
