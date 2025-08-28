cd $HOME/FOMO

. jobs/environment.sh

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced max_cycles=1 n_epochs_per_cycle=100 experiment_name=base_vitb_simclr_imbalanced

