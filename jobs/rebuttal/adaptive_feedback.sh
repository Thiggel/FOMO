#!/bin/bash
#SBATCH --job-name=fomo-feedback
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-14
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Equal total updates and repair budget: isolate whether recomputing the policy
# matters beyond one-shot or frozen-anchor repetition.
conditions=(adaptive static one_shot dense_placebo top_tail)
task="${SLURM_ARRAY_TASK_ID}"; seed="$((task % 3))"; condition="${conditions[$((task / 3))]}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
checkpoint="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"
selection=ood; strategy=mode_window; reuse=adaptive; once=false
case "$condition" in
  static) reuse=static_first_cycle ;;
  one_shot) once=true ;;
  dense_placebo) strategy=dense ;;
  top_tail) strategy=top ;;
esac
run_root="$BASE_CACHE_DIR/rebuttal_runs/adaptive_feedback/$condition/seed_${seed}${run_suffix}"; mkdir -p "$run_root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true max_cycles=6 n_epochs_per_cycle=100 max_steps_per_cycle=1616 \
  train_batch_size=128 val_batch_size=256 num_ood_samples=100 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy="$strategy" selection_reuse_policy="$reuse" repair_once="$once" \
  ood_augmentation=true generation_model=strong_augmentation ood_distance_metric=normalized_l2 \
  additional_data_path="$run_root/generated" experiment_name="rebuttal_feedback_${condition}${run_suffix}"
