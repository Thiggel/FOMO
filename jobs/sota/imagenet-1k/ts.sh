#!/bin/sh
#PBS -q rt_HF
#PBS -l select=4
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/imagenet-1k

python -m experiment \
    model=resnet50 \
    dataset=imagenet1k_alldata \
    ssl=simclr \
    ood_augmentation=false \
    use_temperature_schedule=true \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_imagenet-1k_ts \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3 >& job_logs/imagenet-1k/ts.out
