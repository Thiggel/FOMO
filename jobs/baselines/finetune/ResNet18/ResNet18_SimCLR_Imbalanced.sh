. jobs/environment.sh

python -m experiment \
	--model ResNet18 \
	--imbalance_method power_law_imbalance \
	--no-pretrain \
	--checkpoint ResNet18_SimCLR_Imbalanced-1.ckpt ResNet18_SimCLR_Imbalanced-2.ckpt ResNet18_SimCLR_Imbalanced-3.ckpt
