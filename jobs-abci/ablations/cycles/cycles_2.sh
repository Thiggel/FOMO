#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/cycles

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced max_cycles=2 n_epochs_per_cycle=50 ood_augmentation=true experiment_name=ablations_cycles_cycles_2 train_batch_size=512 use_deepspeed=false >& job_logs/cycles/cycles_2.out
