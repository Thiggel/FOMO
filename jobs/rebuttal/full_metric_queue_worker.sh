#!/bin/bash
# Claim full-metric backfill cells from a shared queue until the queue is
# empty, the GPU is no longer usable, or the worker runs out of walltime.
#
# Workers are deliberately stateless and idempotent: a cell describes itself
# completely, and backfill_full_metrics.sh exits early when a result already
# carries the full suite.  A cell claimed by a worker that dies is therefore
# safe to requeue.
set -uo pipefail

cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR is required}"
. jobs/rebuttal/load_cluster_environment.sh

queue="${FOMO_METRIC_QUEUE:?FOMO_METRIC_QUEUE is required}"
worker="${FOMO_WORKER_NAME:-${SLURM_JOB_ID:-$$}}"
pending="$queue/pending"
running="$queue/running"
completed="$queue/completed"
failed="$queue/failed"
logs="$queue/logs"
mkdir -p "$pending" "$running" "$completed" "$failed" "$logs"

# Stop claiming new cells once too little walltime remains to finish one.
# Alex caps jobs at 24h, so a worker that starts a two-hour evaluation at
# hour 23 would be killed mid-benchmark and waste the whole slot.
reserve_seconds="${FOMO_QUEUE_RESERVE_SECONDS:-9000}"
deadline_seconds="${FOMO_QUEUE_DEADLINE_SECONDS:-0}"
started_at="$SECONDS"

# Evaluation-only work (linear probe + kNN) needs far less memory than the
# training runs this threshold was originally chosen for.  Requiring a nearly
# empty card made workers quit immediately whenever an MPS-sharing neighbour
# held even a few GB.
min_free_gpu_mib="${FOMO_MIN_FREE_GPU_MIB:-12000}"
gpu_wait_attempts="${FOMO_GPU_WAIT_ATTEMPTS:-10}"
gpu_wait_seconds="${FOMO_GPU_WAIT_SECONDS:-60}"

# Concurrent PyTorch workers exhaust a node's shared /tmp; use node-local
# memory instead, which also avoids stale .nfs files on teardown.
#
# Use exactly the path load_cluster_environment.sh derives, because
# backfill_full_metrics.sh sources that script again per cell and would
# otherwise point the run at a second directory.  Keeping two paths meant the
# exit trap below deleted one of them while a run was still using the other,
# which surfaced as tempfile.mkdtemp failing with ENOENT a few seconds into a
# cell -- reliably so for pass_subset, whose loader stages an archive there.
export TMPDIR="/dev/shm/fomo_${USER:-user}_gpu_${worker}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
# Only clean up on a clean exit.  Several workers share a node, and a trap
# that fires while a sibling is mid-cell is one of the ways scratch space has
# disappeared under a running evaluation.
trap 'rm -rf "$TMPDIR" 2>/dev/null || true' EXIT

have_time_for_another_cell() {
    [[ "$deadline_seconds" -gt 0 ]] || return 0
    local elapsed=$(( SECONDS - started_at ))
    (( deadline_seconds - elapsed > reserve_seconds ))
}

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d '[:space:]'
}

# Wait for a transiently busy card rather than abandoning the allocation: the
# GPU is already reserved for this job, so giving up here just wastes it.
gpu_is_usable() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local attempt free_mib
    for (( attempt = 1; attempt <= gpu_wait_attempts; attempt++ )); do
        free_mib="$(gpu_free_mib)"
        [[ "$free_mib" =~ ^[0-9]+$ ]] || return 0
        (( free_mib >= min_free_gpu_mib )) && return 0
        echo "WAIT worker=$worker free_mib=$free_mib required=$min_free_gpu_mib" \
             "attempt=$attempt/$gpu_wait_attempts"
        sleep "$gpu_wait_seconds"
    done
    return 1
}

claim_cell() {
    local candidate name target
    shopt -s nullglob
    local candidates=("$pending"/*.json)
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
        name="$(basename "$candidate")"
        target="$running/$name.$worker"
        # mv within one filesystem is atomic, so exactly one worker wins.
        if mv "$candidate" "$target" 2>/dev/null; then
            printf '%s' "$target"
            return 0
        fi
    done
    return 1
}

while true; do
    if ! have_time_for_another_cell; then
        echo "STOP worker=$worker reason=walltime_reserve"
        break
    fi
    if ! gpu_is_usable; then
        echo "STOP worker=$worker reason=gpu_busy min_free_mib=$min_free_gpu_mib"
        break
    fi

    claimed="$(claim_cell)" || { echo "STOP worker=$worker reason=queue_empty"; break; }
    name="$(basename "$claimed" ".$worker")"

    # Recreate the scratch directory before every cell.  Something on these
    # nodes removes /dev/shm entries out from under a running job -- a sibling
    # worker's exit trap, or a node cleanup sweep -- and the failure surfaces
    # far from the cause, as tempfile or wandb dying on a missing path several
    # benchmarks in.  Asserting it here is cheap and makes the run independent
    # of whatever else touches /dev/shm.
    mkdir -p "$TMPDIR"

    experiment="$(jq -r .experiment "$claimed")"
    seed="$(jq -r .seed "$claimed")"
    dataset_dir="$(jq -r .dataset_dir "$claimed")"
    dataset_config="$(jq -r .dataset_config "$claimed")"
    model_config="$(jq -r .model_config "$claimed")"
    ssl_config="$(jq -r .ssl_config "$claimed")"

    log="$logs/${name%.json}.$worker.log"
    {
        printf 'START %s worker=%s host=%s cuda=%s\n' \
            "$(date --iso-8601=seconds)" "$worker" "$(hostname)" \
            "${CUDA_VISIBLE_DEVICES:-unset}"
        printf 'CELL experiment=%s seed=%s model=%s ssl=%s dataset=%s\n' \
            "$experiment" "$seed" "$model_config" "$ssl_config" "$dataset_config"

        if SOURCE_EXPERIMENT="$experiment" \
           SOURCE_SEED="$seed" \
           SOURCE_DATASET_DIR="$dataset_dir" \
           SOURCE_DATASET_CONFIG="$dataset_config" \
           SOURCE_MODEL_CONFIG="$model_config" \
           SOURCE_SSL_CONFIG="$ssl_config" \
           bash jobs/rebuttal/backfill_full_metrics.sh; then
            printf 'END %s status=0\n' "$(date --iso-8601=seconds)"
            mv "$claimed" "$completed/$name.$worker"
        else
            status="$?"
            printf 'END %s status=%s\n' "$(date --iso-8601=seconds)" "$status"
            mv "$claimed" "$failed/$name.$worker"
        fi
    } >> "$log" 2>&1

    echo "DONE worker=$worker cell=$name"
done
