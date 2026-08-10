#!/bin/bash
# Run the downstream cells originally queued behind the long one-shot policy
# experiment. Completed cells are skipped by the task scripts, so the original
# supervisors can finish later without duplicating results.

set -uo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

worker="${QUEUE_WORKER:?QUEUE_WORKER must be 0, 1, or 2}"
(( worker >= 0 && worker <= 2 )) || exit 2
workers=9
suffix="${FOMO_RUN_SUFFIX:-_iclr2027_full_v2}"
log_root="$BASE_CACHE_DIR/logs/iclr2027_remaining"
marker_root="$BASE_CACHE_DIR/rebuttal_runs/queue_markers/${suffix#_}"
mkdir -p "$log_root" "$marker_root"

run_task() {
  stage="$1"
  script="$2"
  task="$3"
  log="$log_root/${stage}_task${task}${suffix}.log"
  echo "[$(date --iso-8601=seconds)] START $stage task=$task worker=$worker-recovery" >> "$log"
  if SLURM_ARRAY_TASK_ID="$task" FOMO_RUN_SUFFIX="$suffix" bash "$script" >> "$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] END $stage task=$task status=$status" >> "$log"
  return "$status"
}

for task in $(seq 0 26); do
  (( task % workers == worker )) || continue
  run_task percentile jobs/rebuttal/percentile_utility.sh "$task"
done
touch "$marker_root/percentile_worker_${worker}.done"

for task in $(seq 0 77); do
  (( task % workers == worker )) || continue
  run_task selector jobs/rebuttal/selector_sweep.sh "$task"
done

run_task scale100k jobs/rebuttal/scale_sweep.sh "$((36 + worker))"

if (( worker == 0 )); then
  while :; do
    complete=0
    for index in $(seq 0 8); do
      [[ -f "$marker_root/percentile_worker_${index}.done" ]] && complete=$((complete + 1))
    done
    (( complete == workers )) && break
    sleep 60
  done
  run_task repair_fidelity jobs/rebuttal/repair_fidelity.sh 0
fi

touch "$marker_root/all_worker_${worker}.done"
