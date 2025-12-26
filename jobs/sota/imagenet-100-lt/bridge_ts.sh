#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/imagenet-100-lt

torchrun --standalone --nproc_per_node=8 -m experiment \
    model=resnet50 \
    dataset=imagenet100_imbalanced \
    ssl=simclr \
    ood_augmentation=true \
    use_temperature_schedule=true \
    max_cycles=8 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_imagenet-100-lt_bridge-ts \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3 >& job_logs/imagenet-100-lt/bridge_ts.out
