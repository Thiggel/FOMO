. jobs/environment.sh

python -m experiment \
	--imbalance_method linearly_increasing \
	--no-pretrain \
	--checkpoint ResNet18_SimCLR_Imbalanced-1.ckpt ResNet18_SimCLR_Imbalanced-2.ckpt ResNet18_SimCLR_Imbalanced-3.ckpt
