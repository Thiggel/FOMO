cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=imagenet100_imbalanced max_cycles=1 n_epochs_per_cycle=800 use_temperature_schedule=true ood_augmentation=true experiment_name=sota_newmethod_ts
