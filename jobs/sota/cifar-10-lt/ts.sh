cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=cifar10_imbalanced num_cycles=1 total_epochs=800 use_temperature_schedule=true experiment_name=sota_cifar-10-lt_ts
