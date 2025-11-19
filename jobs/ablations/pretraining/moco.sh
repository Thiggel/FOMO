#!/bin/sh
#PBS -q rt_HF
#PBS -l select=4
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations

python -m experiment \
    model=resnet50 \
    ssl=moco \
    dataset=imagenet100_imbalanced \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=ablations_pretraining_moco \
    train_batch_size=512 >& job_logs/ablations/pretraining_moco.out
