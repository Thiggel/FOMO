#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/pretraining

python -m experiment model=vit_base ssl=moco dataset=imagenet1k_imbalanced max_cycles=5 n_epochs_per_cycle=20 ood_augmentation=true experiment_name=ablations_pretraining_moco train_batch_size=512 use_deepspeed=false >& job_logs/pretraining/moco.out
