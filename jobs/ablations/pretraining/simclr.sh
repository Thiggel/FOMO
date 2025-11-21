#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations

torchrun --standalone --nproc_per_node=8 -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=ablations_pretraining_simclr \
    train_batch_size=512 >& job_logs/ablations/pretraining_simclr.out
