#!/bin/bash
#SBATCH --job-name=fomo-branch-source
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

seed="${SLURM_ARRAY_TASK_ID}"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=false \
    num_runs=1 seed="$seed" \
    train_batch_size=128 val_batch_size=256 \
    max_cycles=1 n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
    ood_augmentation=false \
    experiment_name=rebuttal_branch_source

checkpoint_dir="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed"
checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$checkpoint"
printf '%s\n' "$checkpoint" >"$checkpoint_dir/branch_checkpoint.txt"

