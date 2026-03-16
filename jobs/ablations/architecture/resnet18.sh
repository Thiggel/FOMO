#!/bin/sh
#SBATCH --job-name=ablations-architecture-resnet18
#SBATCH --output=job_logs/ablations/architecture/resnet18_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations/architecture

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet18 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=ablations_architecture_resnet18 \
    num_runs=3 \
    train_batch_size=512
