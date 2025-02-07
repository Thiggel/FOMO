. jobs/environment.sh

srun python -m experiment \
	--model ViTSmall \
	--imbalance-method power_law_imbalance \
	--max-cycles 5 \
	--ood-augmentation \
	--n-epochs-per-cycle 20 \
	--pct-ood 0.15 \
	--crop-size 224 \
  --ssl-method MoCo \
  --batch-size 335 \
  --experiment-name "NewMethod_ViTSmall_MoCo_Imbalanced"
