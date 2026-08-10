#!/bin/bash
#SBATCH --job-name=fomo-sd3-distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
seed="$((SLURM_ARRAY_TASK_ID%3))"; bridge=false; [[ $SLURM_ARRAY_TASK_ID -ge 3 ]] && bridge=true
cycles=1; steps=9700; [[ "$bridge" == true ]] && { cycles=2; steps=4850; }
root="$BASE_CACHE_DIR/rebuttal_runs/sd3_distill/$bridge/seed_$seed"; mkdir -p "$root"
# This is SD3 VAE-latent distillation on original source images.  The frozen
# targets are precomputed once, so no SD3 component is evaluated every SSL step.
cache="$BASE_CACHE_DIR/sd3_teacher_cache/seed_${seed}.pt"; test -f "$cache"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true finetune_benchmark_suite=paper_full num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" train_batch_size=128 val_batch_size=256 external_diffusion_teacher=true external_diffusion_teacher_cache="$cache" external_teacher_weight=0.1 ood_augmentation="$bridge" generation_model=stable_diffusion_3 additional_data_path="$root/generated" experiment_name="rebuttal_sd3_vae_distill_$bridge"
