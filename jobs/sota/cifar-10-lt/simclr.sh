cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=cifar10_imbalanced max_cycles=1 n_epochs_per_cycle=800 experiment_name=sota_cifar-10-lt_simclr
