#!/bin/sh
#PBS -q rt_HF
#PBS -l select=4
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/diffusiondb-subset

python -m experiment \
    model=resnet50 \
    dataset=diffusiondb_subset \
    ssl=sdclr \
    ood_augmentation=true \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_diffusiondb-subset_bridge-sdclr \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3 >& job_logs/diffusiondb-subset/bridge_sdclr.out
