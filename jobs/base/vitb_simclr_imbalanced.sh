cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced num_cycles=1 total_epochs=100 experiment_name=base_vitb_simclr_imbalanced

