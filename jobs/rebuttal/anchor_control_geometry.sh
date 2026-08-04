#!/bin/bash
# Post-hoc local treatment analysis requested by reviewer 2x5B.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
base="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
bridge="$CHECKPOINT_ROOT_DIR/deadline_mc3e60_resnet_bridge_${seed}_gruenau_retry1/clane9_imagenet-100/seed_${seed}/last.ckpt"
run_root="$BASE_CACHE_DIR/rebuttal_runs/deadline_mc3e60_resnet_bridge_${seed}_gruenau_retry1/generated"
selections=("$run_root"/ood_diagnostics/*/selection.npz)
args=(
  --base "$base"
  --bridge "$bridge"
  --selections "${selections[@]}"
  --seed "$seed"
  --output "$BASE_CACHE_DIR/rebuttal_analysis/anchor_control_geometry/seed_${seed}.json"
)
no_repair="$CHECKPOINT_ROOT_DIR/rebuttal_fullpolicy3e60_no_repair_${seed}_gruenau_retry1/clane9_imagenet-100/seed_${seed}/last.ckpt"
if [[ -s "$no_repair" ]]; then
  args+=(--no-repair "$no_repair")
fi
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python3 paper_work/analysis/anchor_control_geometry.py "${args[@]}"
