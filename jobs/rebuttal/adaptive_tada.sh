#!/bin/bash
#SBATCH --job-name=fomo-adaptive-tada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Adaptive counterpart to adaptive_feedback.sh.  The current encoder
# recomputes TADA-style per-example SSL difficulty before every repair stage.
# We deliberately use the same inexpensive repair operator, update count, and
# total repair volume as adaptive BRIDGE here; the one-stage prior_controls.sh
# experiment already compares TADA-style and BRIDGE selection with matched SD3.
seed="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
checkpoint="$(
    cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt"
)"
run_root="$BASE_CACHE_DIR/rebuttal_runs/adaptive_tada/seed_${seed}${run_suffix}"
mkdir -p "$run_root"

python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
    max_cycles=6 n_epochs_per_cycle=100 max_steps_per_cycle=1616 \
    train_batch_size=128 val_batch_size=256 \
    num_ood_samples=100 num_generations_per_ood_sample=5 \
    sample_selection=early_loss selection_reuse_policy=adaptive repair_once=false \
    ood_augmentation=true generation_model=strong_augmentation \
    ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_adaptive_tada${run_suffix}"
