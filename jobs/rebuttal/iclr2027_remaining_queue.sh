#!/bin/bash
# One sequential queue per GPU. Tasks are distributed round-robin across nine
# workers so every reported cell retains three independent seeds and writes a
# result.json with all downstream benchmarks.

set -uo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

worker="${QUEUE_WORKER:?QUEUE_WORKER must be in 0..8}"
workers=9
suffix="${FOMO_RUN_SUFFIX:-_iclr2027_full}"
log_root="$BASE_CACHE_DIR/logs/iclr2027_remaining"
marker_root="$BASE_CACHE_DIR/rebuttal_runs/queue_markers/${suffix#_}"
mkdir -p "$log_root" "$marker_root"

run_task() {
  stage="$1"
  script="$2"
  task="$3"
  log="$log_root/${stage}_task${task}${suffix}.log"
  echo "[$(date --iso-8601=seconds)] START $stage task=$task worker=$worker" >> "$log"
  if SLURM_ARRAY_TASK_ID="$task" FOMO_RUN_SUFFIX="$suffix" bash "$script" >> "$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] END $stage task=$task status=$status" >> "$log"
  return "$status"
}

for task in $(seq 0 11); do
  (( task % workers == worker )) || continue
  run_task full_policy jobs/rebuttal/full_policy_5c100.sh "$task"
done

for task in $(seq 0 26); do
  (( task % workers == worker )) || continue
  run_task percentile jobs/rebuttal/percentile_utility.sh "$task"
done
touch "$marker_root/percentile_worker_${worker}.done"

if (( worker == 0 )); then
  while :; do
    complete=0
    for index in $(seq 0 8); do
      test -f "$marker_root/percentile_worker_${index}.done" && complete=$((complete + 1))
    done
    (( complete == workers )) && break
    sleep 60
  done
  run_task repair_fidelity jobs/rebuttal/repair_fidelity.sh 0
fi

for task in $(seq 0 77); do
  (( task % workers == worker )) || continue
  run_task selector jobs/rebuttal/selector_sweep.sh "$task"
done

for task in 36 37 38 39 40 41; do
  (( task % workers == worker )) || continue
  run_task scale100k jobs/rebuttal/scale_sweep.sh "$task"
done

touch "$marker_root/all_worker_${worker}.done"
