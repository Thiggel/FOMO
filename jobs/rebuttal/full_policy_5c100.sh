#!/bin/bash
# Exact main-schedule comparison of continued training, adaptive selection,
# a frozen first-cycle selector, and one-shot repair. Each arm receives five
# 100-epoch post-branch stages capped at the same number of optimizer updates.
# Repair arms add 12,500 images in total.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(no_repair adaptive static one_shot)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]:?Unknown full-policy task $task}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"

augment=true
reuse=adaptive
repair_once=false
generations=5
case "$condition" in
  no_repair) augment=false ;;
  adaptive) ;;
  static) reuse=static_first_cycle ;;
  one_shot) repair_once=true; generations=25 ;;
  *) exit 2 ;;
esac

suffix="${FOMO_RUN_SUFFIX:-}"
tag="iclr_fullpolicy5e100_${condition}_${seed}${suffix}"
root="$BASE_CACHE_DIR/rebuttal_runs/$tag"
mkdir -p "$root"

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=6 \
  n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample="$generations" \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy="$reuse" repair_once="$repair_once" \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$root/generated" experiment_name="$tag"
