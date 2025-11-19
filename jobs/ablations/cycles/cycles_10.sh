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
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=10 \
    n_epochs_per_cycle=80 \
    ood_augmentation=true \
    experiment_name=ablations_cycles_10 \
    train_batch_size=512 >& job_logs/ablations/cycles_10.out
