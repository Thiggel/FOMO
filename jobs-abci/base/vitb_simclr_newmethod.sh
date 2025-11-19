#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/base

python -m experiment model=vit_base ssl=simclr dataset=imagenet1k_imbalanced num_cycles=5 total_epochs=100 ood_augmentation=true experiment_name=base_vitb_simclr_newmethod train_batch_size=512 use_deepspeed=false >& job_logs/base/vitb_simclr_newmethod.out
