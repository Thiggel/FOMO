cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=imagenet100_imbalanced num_cycles=8 total_epochs=800 use_temperature_schedule=true experiment_name=sota_imagenet-100-lt_ts
