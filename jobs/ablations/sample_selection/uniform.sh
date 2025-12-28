#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations

torchrun --standalone --nproc_per_node=1 -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    sample_selection=uniform \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=ablations_sample-selection_uniform \
    num_runs=3 \
    train_batch_size=512 >& job_logs/ablations/sample-selection_uniform.out
