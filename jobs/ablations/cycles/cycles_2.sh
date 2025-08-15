cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced max_cycles=2 n_epochs_per_cycle=50 ood_augmentation=true experiment_name=ablations_cycles_cycles_2

