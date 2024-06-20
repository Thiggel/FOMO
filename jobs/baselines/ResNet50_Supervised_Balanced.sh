. jobs/environment.sh

python -m experiment \
  --ssl_method Supervised \
	--imbalance_method no_imbalance \
	--max_cycles 1 \
	--n_epochs_per_cycle 1 \
  --no-logger
