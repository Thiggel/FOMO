cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=cifar100_imbalanced num_cycles=1 total_epochs=800 experiment_name=sota_cifar-100-lt_simclr
