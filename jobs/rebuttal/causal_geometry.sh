#!/bin/bash
#SBATCH --job-name=fomo-causal-geometry
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh

# Geometry-only companion to the finished factorial: identical branch point,
# update budget, and repair volume, but saves fixed-original-panel snapshots
# after every training stage for the causal density plots.
conditions=(no_repair uniform_sd3 mode_sd3 top_sd3)
task="$SLURM_ARRAY_TASK_ID"; seed="$((task % 3))"; condition="${conditions[$((task / 3))]}"
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/last.ckpt"
selection=ood; strategy=mode_window; augment=true
case "$condition" in
  no_repair) augment=false ;;
  uniform_sd3) selection=random ;;
  top_sd3) strategy=top ;;
esac
root="$BASE_CACHE_DIR/rebuttal_runs/causal_geometry/$condition/seed_$seed"; mkdir -p "$root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=false \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
  max_cycles=2 n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
  train_batch_size=128 val_batch_size=256 num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy="$strategy" ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 representation_diagnostics_each_cycle=true \
  additional_data_path="$root/generated" experiment_name="rebuttal_causal_geometry_${condition}"
