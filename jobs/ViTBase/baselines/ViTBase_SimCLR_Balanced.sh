. jobs/environment.sh

srun python -m experiment \
	--model-name ViTBase \
	--imbalance-method no_imbalance \
	--max-cycles 1 \
	--n-epochs-per-cycle 100 \
  --train-batch-size 1024 \
  --val-batch-size 1024 \
  --experiment-name "Baseline_ViTBase_SimCLR_Balanced"

