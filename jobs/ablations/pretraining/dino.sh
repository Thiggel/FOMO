#!/bin/sh
#SBATCH --job-name=ablations-pretraining-dino
#SBATCH --output=job_logs/ablations/pretraining/dino_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh
export NPROC_PER_NODE=4

mkdir -p job_logs/ablations/pretraining

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    ssl=dino \
    dataset=imagenet100_imbalanced \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    sd3_batch_size=12 \
    experiment_name=ablations_pretraining_dino \
    num_runs=3 \
    train_batch_size=256 \
    grad_acc_steps=2
