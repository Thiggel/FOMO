#!/bin/bash
# Recover the cells assigned to queue worker 8 after its original 23 GB GPU
# process stopped. Keep the original suffix so these runs fill the missing
# third-seed cells rather than creating a second result family.

set -uo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

suffix="${FOMO_RUN_SUFFIX:-_iclr2027_full_v2}"
log_root="$BASE_CACHE_DIR/logs/iclr2027_remaining"
marker_root="$BASE_CACHE_DIR/rebuttal_runs/queue_markers/${suffix#_}"
mkdir -p "$log_root" "$marker_root"

run_task() {
  stage="$1"
  script="$2"
  task="$3"
  log="$log_root/${stage}_task${task}${suffix}.log"
  echo "[$(date --iso-8601=seconds)] START $stage task=$task worker=8-recovery" >> "$log"
  if SLURM_ARRAY_TASK_ID="$task" FOMO_RUN_SUFFIX="$suffix" bash "$script" >> "$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] END $stage task=$task status=$status" >> "$log"
  return "$status"
}

for task in 8 17 26; do
  run_task percentile jobs/rebuttal/percentile_utility.sh "$task"
done
touch "$marker_root/percentile_worker_8.done"

for task in 8 17 26 35 44 53 62 71; do
  run_task selector jobs/rebuttal/selector_sweep.sh "$task"
done

touch "$marker_root/all_worker_8.done"
