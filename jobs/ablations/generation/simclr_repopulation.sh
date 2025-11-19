cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced remove_diffusion=true num_cycles=5 total_epochs=100 ood_augmentation=true experiment_name=ablations_generation_simclr_repopulation

