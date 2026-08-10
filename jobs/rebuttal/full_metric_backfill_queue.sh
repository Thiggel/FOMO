#!/bin/bash
# Shard an evaluation-only backfill manifest over one or more GPUs.

set -uo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

manifest="${BACKFILL_MANIFEST:?BACKFILL_MANIFEST is required}"
worker="${BACKFILL_WORKER:?BACKFILL_WORKER is required}"
workers="${BACKFILL_WORKERS:?BACKFILL_WORKERS is required}"
log_root="$BASE_CACHE_DIR/logs/full_metric_backfill"
mkdir -p "$log_root"

index=0
while IFS=$'\t' read -r experiment seed; do
  [[ -n "$experiment" ]] || continue
  if (( index % workers != worker )); then
    index=$((index + 1))
    continue
  fi

  log="$log_root/${experiment}_seed${seed}.log"
  echo "[$(date --iso-8601=seconds)] START index=$index" >> "$log"
  if SOURCE_EXPERIMENT="$experiment" SOURCE_SEED="$seed" \
      bash jobs/rebuttal/backfill_full_metrics.sh >> "$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "[$(date --iso-8601=seconds)] END status=$status" >> "$log"
  index=$((index + 1))
done < "$manifest"
