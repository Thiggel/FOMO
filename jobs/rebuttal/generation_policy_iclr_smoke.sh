#!/bin/bash
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
seed=0
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
run_tag="iclr_generation_policy_vlm_t2i_smoke${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"
python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=false num_runs=1 seed="$seed" \
  checkpoint="$checkpoint" skip_initial_training=true \
  max_cycles=2 n_epochs_per_cycle=1 limit_train_batches=1 \
  train_batch_size=8 val_batch_size=16 num_ood_samples=2 \
  num_generations_per_ood_sample=1 sample_selection=ood \
  ood_selection_strategy=mode_window ood_distance_metric=normalized_l2 \
  ood_augmentation=true generation_model=stable_diffusion_3_t2i \
  sd3_batch_size=1 additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
