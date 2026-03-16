#!/bin/sh
#SBATCH --job-name=baseline-newmethod-imbalanced
#SBATCH --output=job_logs/baseline/newmethod_imbalanced_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/baseline

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=baseline_imagenet-100-lt_newmethod \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3
