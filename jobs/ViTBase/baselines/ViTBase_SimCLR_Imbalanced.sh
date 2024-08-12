. jobs/environment.sh

python -m experiment \
	--model_name ViTBase \
	--max_cycles 1 \
	--n_epochs_per_cycle 100 \
  --crop_size 224 \
  --experiment_name "Baseline_ViTBase_SimCLR_Imbalanced"
