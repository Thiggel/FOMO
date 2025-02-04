. jobs/environment.sh

python -m experiment \
	--model ResNet50 \
	--imbalance-method power_law_imbalance \
  --num-runs 3 \
	--max-cycles 5 \
	--ood-augmentation \
	--no-use-ood \
	--n-epochs-per-cycle 20 \
  --batch-size 1024 \
  --experiment-name "ResNet50_UniformGeneration_1k"
