#!/bin/bash
#SBATCH --job-name=fomo-percentile
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-26
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

bins=(q00_25 q25_50 q50_75 q75_85 q85_90 q90_95 q95_97 q97_99 q99_100)
task="${SLURM_ARRAY_TASK_ID}"; seed="$((task % 3))"; condition="${bins[$((task / 3))]}"
range='[0.0,0.25]'
case "$condition" in
 q25_50) range='[0.25,0.50]' ;; q50_75) range='[0.50,0.75]' ;; q75_85) range='[0.75,0.85]' ;;
 q85_90) range='[0.85,0.90]' ;; q90_95) range='[0.90,0.95]' ;; q95_97) range='[0.95,0.97]' ;;
 q97_99) range='[0.97,0.99]' ;; q99_100) range='[0.99,1.0]' ;;
esac
checkpoint="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"
run_suffix="${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/percentile_utility/$condition/seed_${seed}${run_suffix}"; mkdir -p "$run_root"
result="$CHECKPOINT_ROOT_DIR/rebuttal_percentile_${condition}${run_suffix}/clane9_imagenet-100/seed_${seed}/result.json"
if fomo_has_full_metric_suite "$result"; then
  echo "Complete result already exists at $result; skipping."
  exit 0
fi
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true max_cycles=2 n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
  train_batch_size=128 val_batch_size=256 ood_percentile_bin="$range" ood_augmentation=true generation_model=stable_diffusion_3 \
  ood_distance_metric=normalized_l2 additional_data_path="$run_root/generated" experiment_name="rebuttal_percentile_${condition}${run_suffix}"
