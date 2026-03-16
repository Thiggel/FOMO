#!/bin/sh
#SBATCH --job-name=sota-diffusiondb-subset-bridge-sdclr
#SBATCH --output=job_logs/sota/diffusiondb-subset/bridge_sdclr_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/sota/diffusiondb-subset

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    dataset=diffusiondb_subset \
    splits='[1.0,0.0,0.0]' \
    ssl=sdclr \
    ood_augmentation=true \
    sd3_batch_size=12 \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    experiment_name=sota_diffusiondb-subset_bridge-sdclr \
    train_batch_size=512 \
    log_class_dist=true \
    log_generated_samples=true \
    log_tsne=true \
    num_runs=3
