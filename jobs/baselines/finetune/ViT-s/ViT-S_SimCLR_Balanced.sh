. jobs/environment.sh

python -m experiment \
	--model_name ViTSmall \
	--imbalance_method no_imbalance \
	--no-pretrain \
	--wandb_checkpoint organize/FOMO/run-qltitsa6-history:v2 organize/FOMO/run-nlagvqek-history:v0
