. jobs/environment.sh

python -m experiment \
	--imbalance_method no_imbalance \
	--no-pretrain \
	--checkpoint ResNet18_SimCLR_Balanced-1.ckpt ResNet18_SimCLR_Balanced-2.ckpt ResNet18_SimCLR_Balanced-3.ckpt
