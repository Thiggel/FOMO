#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/diffusiondb-subset

torchrun --standalone --nproc_per_node=8 -m experiment \
    model=resnet50 \
    dataset=diffusiondb_subset \
    ssl=simclr \
    ood_augmentation=false \
    use_temperature_schedule=true \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_diffusiondb-subset_ts \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3 >& job_logs/diffusiondb-subset/ts.out
