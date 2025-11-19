cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=moco dataset=imagenet1k_imbalanced num_cycles=5 total_epochs=100 ood_augmentation=true experiment_name=ablations_pretraining_moco

