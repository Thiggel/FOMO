#!/bin/bash
# Minimal end-to-end smoke for the VLM-selected text-to-image control.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
seed="${SLURM_ARRAY_TASK_ID:-0}"
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
run_tag="iclr_aide_ssl_smoke_${seed}${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=false \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=2 n_epochs_per_cycle=0 \
  max_steps_per_cycle=1 train_batch_size=128 val_batch_size=256 \
  num_ood_samples=2 num_generations_per_ood_sample=1 \
  sample_selection=ood selection_encoder=clip \
  ood_selection_strategy=cluster_inverse ood_distance_metric=normalized_l2 \
  selection_reuse_policy=adaptive ood_augmentation=true \
  generation_model=stable_diffusion_3_t2i \
  representation_diagnostics_each_cycle=false \
  additional_data_path="$run_root/generated" experiment_name="$run_tag"
