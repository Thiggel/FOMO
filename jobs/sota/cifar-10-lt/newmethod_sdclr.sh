cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=sdclr dataset=cifar10_imbalanced max_cycles=8 n_epochs_per_cycle=100 ood_augmentation=true experiment_name=sota_cifar-10-lt_newmethod_sdclr
