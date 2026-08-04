#!/bin/bash

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
base="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
run_root="$BASE_CACHE_DIR/rebuttal_runs/deadline_mc3e60_resnet_bridge_${seed}_gruenau_retry1"
output="$BASE_CACHE_DIR/rebuttal_analysis/fixed_support_injection/seed_${seed}.json"
test -s "$base"
test -d "$run_root/generated/repair_manifests"
mkdir -p "$(dirname "$output")"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python paper_work/analysis/fixed_support_injection.py \
  --base "$base" --run-root "$run_root" --seed "$seed" --k 20 --output "$output"
