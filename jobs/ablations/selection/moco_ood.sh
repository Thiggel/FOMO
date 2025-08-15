cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=moco dataset=imagenet1k_imbalanced sample_selection=ood max_cycles=5 n_epochs_per_cycle=20 ood_augmentation=true experiment_name=ablations_selection_moco_ood

