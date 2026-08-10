#!/bin/bash
# Matched generation-policy comparison motivated by the NeurIPS reviews.
# All arms use the same sparse anchors, repair volume, and SSL update budget.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(sdedit vlm_t2i)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]:?Unknown generation-policy task $task}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
generator=stable_diffusion_3
if [[ "$condition" == vlm_t2i ]]; then
  generator=stable_diffusion_3_t2i
fi

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="iclr_generation_policy_${condition}_${seed}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=4 \
  n_epochs_per_cycle=60 max_steps_per_cycle=4850 \
  train_batch_size=128 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation=true generation_model="$generator" \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
