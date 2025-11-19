cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=sdclr dataset=cifar100_imbalanced num_cycles=8 total_epochs=800 ood_augmentation=true experiment_name=sota_cifar-100-lt_newmethod_sdclr
