#!/bin/bash
#SBATCH --job-name=fomo-adaptive-bridge-tada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Adaptive hybrid matched to adaptive_feedback.sh and adaptive_tada.sh.
# Both signals are recomputed before every repair while total updates and
# generated variants remain fixed.
seed="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
checkpoint="$(
    cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt"
)"
run_root="$BASE_CACHE_DIR/rebuttal_runs/adaptive_bridge_tada/seed_${seed}${run_suffix}"
mkdir -p "$run_root"

python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
    max_cycles=6 n_epochs_per_cycle=100 max_steps_per_cycle=1616 \
    train_batch_size=128 val_batch_size=256 \
    num_ood_samples=100 num_generations_per_ood_sample=5 \
    sample_selection=bridge_tada ood_selection_strategy=mode_window \
    selection_reuse_policy=adaptive repair_once=false \
    ood_augmentation=true generation_model=strong_augmentation \
    ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_adaptive_bridge_tada${run_suffix}"
