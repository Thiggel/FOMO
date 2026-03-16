#!/bin/sh
#SBATCH --job-name=sota-cifar-100-lt-simclr
#SBATCH --output=job_logs/sota/cifar-100-lt/simclr_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/sota/cifar-100-lt

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    dataset=cifar100_imbalanced \
    ssl=simclr \
    ood_augmentation=false \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_cifar-100-lt_simclr \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3
