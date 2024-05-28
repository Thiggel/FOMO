. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--imbalance_method linearly_increasing \
	--no-pretrain \
	--wandb_checkpoint organize/FOMO/run-0kx1fow2-history:v0 organize/FOMO/run-kh9tfc5y-history:v0
