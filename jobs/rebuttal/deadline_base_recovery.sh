#!/bin/bash
# Resume the three ResNet source-only controls after their first completed
# 60-epoch stage.  The original attempts exposed the fixed-panel diagnostic's
# inference-mode/DataLoader interaction before stage two.

set -euo pipefail
export FOMO_NUM_WORKERS=0
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
previous="deadline_mc3e60_resnet_base_${seed}${FOMO_RUN_SUFFIX:-}"
checkpoint="$CHECKPOINT_ROOT_DIR/$previous/clane9_imagenet-100/seed_${seed}/last.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  echo "Missing completed stage-one checkpoint: $checkpoint" >&2
  exit 3
fi

run_tag="${previous}_recovery"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  max_cycles=2 n_epochs_per_cycle=60 max_steps_per_cycle=4850 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  ood_augmentation=false representation_diagnostics_each_cycle=true \
  additional_data_path="$run_root/generated" experiment_name="$run_tag"
