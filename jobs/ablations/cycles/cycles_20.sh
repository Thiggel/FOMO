#!/bin/sh
#SBATCH --job-name=ablations-cycles-cycles-20
#SBATCH --output=job_logs/ablations/cycles/cycles_20_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations/cycles

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=20 \
    n_epochs_per_cycle=25 \
    num_ood_samples=125 \
    ood_augmentation=true \
    experiment_name=ablations_cycles_20 \
    num_runs=3 \
    train_batch_size=512
