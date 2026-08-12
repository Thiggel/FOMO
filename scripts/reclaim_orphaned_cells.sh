#!/bin/bash
# Return cells to the queue when the worker holding them is gone.
#
# A worker that is cancelled or killed leaves its claim behind in running/,
# where nothing will ever pick it up again.  This returns those claims to
# pending/ while leaving genuinely active claims alone.
#
# Liveness is decided by the log file's modification time, not by parsing
# Slurm job ids out of worker names: squeue's %a field is the *account*, not
# the array task id, and using it silently marks every worker dead, which
# requeues cells that are still being written.  Two workers evaluating one
# cell would race on the same result.json.

set -uo pipefail

queue="${FOMO_METRIC_QUEUE:-/vol/home-vol2/ml/laitenbf/FOMO_runtime/full_metric_queue}"
# A live worker appends to its cell log continuously; anything untouched for
# this long has lost its worker.
stale_seconds="${FOMO_RECLAIM_STALE_SECONDS:-1800}"
dry_run="${1:-}"

now="$(date +%s)"
reclaimed=0
active=0

for claim in "$queue"/running/*; do
    [[ -e "$claim" ]] || continue
    name="$(basename "$claim")"
    cell="${name%%.json.*}.json"
    worker="${name##*.json.}"
    log="$queue/logs/${cell%.json}.${worker}.log"

    if [[ -e "$log" ]]; then
        age=$(( now - $(date -r "$log" +%s) ))
        if (( age < stale_seconds )); then
            active=$(( active + 1 ))
            continue
        fi
        reason="log idle ${age}s"
    else
        reason="no log"
    fi

    if [[ "$dry_run" == "--dry-run" ]]; then
        echo "WOULD RECLAIM $cell (worker $worker, $reason)"
    else
        mv "$claim" "$queue/pending/$cell" && echo "RECLAIMED $cell (worker $worker, $reason)"
    fi
    reclaimed=$(( reclaimed + 1 ))
done

echo "active=$active reclaimed=$reclaimed"
