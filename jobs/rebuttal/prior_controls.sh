#!/bin/bash
#SBATCH --job-name=fomo-prior-controls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-8

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Closest-prior controls use exactly the same SD3 repair budget.  `early_loss`
# uses per-example symmetric SSL InfoNCE difficulty, mirroring TADA's
# learning-dynamics signal without exposing labels to selection.
conditions=(mode_window early_loss cluster_inverse)
task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
checkpoint="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"

sample_selection=ood
strategy=mode_window
case "$condition" in
    early_loss) sample_selection=early_loss ;;
    cluster_inverse) strategy=cluster_inverse ;;
esac
run_suffix="${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/prior_controls/$condition/seed_${seed}${run_suffix}"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false \
    pretrain=true finetune=true \
    finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
    max_cycles=2 n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
    train_batch_size=128 val_batch_size=256 num_ood_samples=500 num_generations_per_ood_sample=5 \
    sample_selection="$sample_selection" ood_selection_strategy="$strategy" \
    ood_augmentation=true generation_model=stable_diffusion_3 ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" experiment_name="rebuttal_prior_${condition}${run_suffix}"
