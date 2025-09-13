#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/base

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced max_cycles=1 n_epochs_per_cycle=100 experiment_name=base_vitb_simclr_imbalanced train_batch_size=512 use_deepspeed=false >& job_logs/base/vitb_simclr_imbalanced.out
