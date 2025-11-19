#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/pretraining

python -m experiment model=vit_base ssl=dino dataset=imagenet1k_imbalanced max_cycles=5 n_epochs_per_cycle=20 ood_augmentation=true experiment_name=ablations_pretraining_dino train_batch_size=512 >& job_logs/pretraining/dino.out
