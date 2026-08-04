#!/bin/bash
set -euo pipefail

repo="${FOMO_REPO_DIR:-/vol/home-vol2/ml/laitenbf/FOMO}"
runtime="${BASE_CACHE_DIR:-/vol/home-vol2/ml/laitenbf/FOMO_runtime}"
queue="${FOMO_RETRY_QUEUE:-$runtime/gruenau_retry_20260725}"
retry_suffix="${FOMO_RUN_SUFFIX:-_gruenau_retry1}"

enqueue() {
    local class="$1" priority="$2" script="$3" task="$4"
    local stem="${priority}__${script%.sh}__${task}"
    local state
    for state in pending running completed failed; do
        compgen -G "$queue/$class/$state/${stem}*" >/dev/null && return 0
    done
    : > "$queue/$class/pending/${stem}.task"
}

initialize() {
    local class state
    for class in modern heavy light; do
        for state in pending running completed failed logs; do
            mkdir -p "$queue/$class/$state"
        done
    done

    # First wave: diversify the highest-priority reviewer evidence.
    for task in 4 5; do enqueue modern 00 dinov3_teacher.sh "$task"; done
    for task in 0 1 2; do enqueue modern 01 adaptive_feedback.sh "$task"; done
    for task in 0 1 2; do enqueue modern 02 synthetic_provenance.sh "$task"; done
    for task in 0 1 2; do enqueue modern 03 flux_fairness.sh "$task"; done
    for task in 0 1 2; do enqueue modern 04 oracle_real_restoration.sh "$task"; done

    # Remaining failed or incomplete multi-cycle/generative experiments.
    for task in $(seq 3 14); do enqueue modern 10 adaptive_feedback.sh "$task"; done
    for task in $(seq 3 11); do enqueue modern 11 flux_fairness.sh "$task"; done
    for task in $(seq 24 35) $(seq 39 44); do
        enqueue modern 12 fixed_cycles.sh "$task"
    done
    for task in $(seq 0 10); do enqueue modern 13 ts_interaction.sh "$task"; done
    for task in 0 1 2 6 7 8; do enqueue modern 14 prior_controls.sh "$task"; done

    # Move the large queued sensitivity/scaling arrays off Alex.
    for task in 32 38 64; do
        enqueue heavy 20 selector_sweep.sh "$task"
    done
    for task in $(seq 65 77); do enqueue modern 15 selector_sweep.sh "$task"; done
    for task in 6 7 8 9 11; do enqueue modern 16 compatibility.sh "$task"; done
    for task in $(seq 0 47); do enqueue modern 30 scale_sweep.sh "$task"; done

    # Lightweight retries are safe on the 24-GB RTX 6000 cards.
    for task in 0 1 2 3 4 5 9 10 11 12 13 14; do
        enqueue light 00 lowshot_finetune.sh "$task"
    done
    enqueue light 01 voc_segmentation.sh 3
    enqueue light 02 dinov3_geometry.sh single
    enqueue light 02 repair_fidelity.sh single
    for task in 0 2 3 12 13 16; do
        enqueue light 10 compatibility.sh "$task"
    done

    printf 'Initialized retry queue at %s\n' "$queue"
}

run_task() {
    local script="$1" task="$2"
    export FOMO_CLUSTER=gruenau
    export FOMO_REPO_DIR="$repo"
    export BASE_CACHE_DIR="$runtime"
    export FOMO_RUN_SUFFIX="$retry_suffix"
    export SLURM_ARRAY_TASK_ID="$task"

    case "$script" in
        compatibility.sh)
            export FOMO_RUN_GROUP="compatibility${retry_suffix}"
            export FOMO_EXPERIMENT_PREFIX="rebuttal_compat${retry_suffix}"
            ;;
    esac

    if [[ "$task" == single ]]; then
        unset SLURM_ARRAY_TASK_ID
    fi
    bash "$repo/jobs/rebuttal/$script"
}

worker() {
    local class="${1:?worker class required}"
    local worker_name="${2:?worker name required}"
    local min_free_gpu_mib="${FOMO_MIN_FREE_GPU_MIB:-23000}"
    local pending="$queue/$class/pending"
    local running="$queue/$class/running"
    local completed="$queue/$class/completed"
    local failed="$queue/$class/failed"
    local logs="$queue/$class/logs"
    mkdir -p "$pending" "$running" "$completed" "$failed" "$logs"
    # Several concurrent PyTorch/Hugging Face workers can exhaust a node's
    # small shared /tmp filesystem.  Use the node-local, high-capacity tmpfs;
    # NFS temporary directories leave busy .nfs files during multiprocessing
    # teardown.
    export TMPDIR="/dev/shm/fomo_${USER:-user}_${worker_name}"
    export TEMP="$TMPDIR"
    export TMP="$TMPDIR"
    mkdir -p "$TMPDIR"

    cd "$repo"
    . jobs/clusters/gruenau_environment.sh

    while true; do
        # A GPU can be claimed by another user after this worker was launched.
        # Do not consume the remainder of the queue with identical OOM failures
        # when there is no longer enough memory for a rebuttal training job.
        if command -v nvidia-smi >/dev/null 2>&1 &&
           [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
            local free_gpu_mib
            free_gpu_mib="$(
                nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
                    --query-gpu=memory.free --format=csv,noheader,nounits |
                    head -n 1 | tr -d '[:space:]'
            )"
            if [[ "$free_gpu_mib" =~ ^[0-9]+$ ]] &&
               (( free_gpu_mib < min_free_gpu_mib )); then
                printf 'STOP %s worker=%s free_gpu_mib=%s required_gpu_mib=%s\n' \
                    "$(date --iso-8601=seconds)" "$worker_name" \
                    "$free_gpu_mib" "$min_free_gpu_mib"
                break
            fi
        fi

        local claimed="" source_file=""
        shopt -s nullglob
        local candidates=("$pending"/*.task)
        shopt -u nullglob
        for source_file in "${candidates[@]}"; do
            local name
            name="$(basename "$source_file" .task)"
            claimed="$running/${name}.${worker_name}.task"
            if mv "$source_file" "$claimed" 2>/dev/null; then
                break
            fi
            claimed=""
        done

        [[ -n "$claimed" ]] || break

        local name rest priority script task log
        name="$(basename "$claimed" ".${worker_name}.task")"
        priority="${name%%__*}"
        rest="${name#*__}"
        script="${rest%%__*}"
        task="${rest#*__}"
        log="$logs/${name}.${worker_name}.log"
        {
            printf 'START %s worker=%s cuda=%s\n' "$(date --iso-8601=seconds)" \
                "$worker_name" "${CUDA_VISIBLE_DEVICES:-unset}"
            if run_task "${script}.sh" "$task"; then
                printf 'SUCCESS %s\n' "$(date --iso-8601=seconds)"
                mv "$claimed" "$completed/${name}.${worker_name}.task"
            else
                status="$?"
                printf 'FAILED %s status=%s\n' "$(date --iso-8601=seconds)" "$status"
                mv "$claimed" "$failed/${name}.${worker_name}.task"
            fi
        } > "$log" 2>&1
    done
}

status() {
    local class state count
    for class in modern heavy light; do
        printf '%s' "$class"
        for state in pending running completed failed; do
            count="$(find "$queue/$class/$state" -maxdepth 1 -type f 2>/dev/null | wc -l)"
            printf ' %s=%s' "$state" "$count"
        done
        printf '\n'
    done
}

reprioritize_48h() {
    # Preserve deferred work for later recovery, but keep it out of the
    # two-day rebuttal critical path.
    local deferred="$queue/deferred_48h"
    local class file task
    for class in modern heavy light; do
        mkdir -p "$deferred/$class"
    done
    shopt -s nullglob
    for file in "$queue/modern/pending"/30__scale_sweep__*.task \
                "$queue/modern/pending"/13__ts_interaction__*.task \
                "$queue/modern/pending"/15__selector_sweep__*.task \
                "$queue/modern/pending"/16__compatibility__*.task \
                "$queue/light/pending"/10__compatibility__*.task; do
        mv "$file" "$deferred/$(basename "$(dirname "$(dirname "$file")")")/"
    done
    for task in 24 25 26 39 40 41 42 43 44; do
        file="$queue/modern/pending/12__fixed_cycles__${task}.task"
        if [[ -e "$file" ]]; then
            mv "$file" "$deferred/modern/"
        fi
    done
    # Existing low-shot retries now contain the corrected label extraction and
    # should run before secondary analyses.
    for file in "$queue/light/pending"/00__lowshot_finetune__*.task; do
        mv "$file" "$queue/light/pending/000__${file##*00__}"
    done
    shopt -u nullglob

    for task in 0 1 2 3 10 11 12 13 14; do
        enqueue light 000 lowshot_finetune.sh "$task"
    done
    for task in 0 1 2; do enqueue light 001 geometry_accuracy_correlation.sh "$task"; done
    for task in $(seq 0 5); do enqueue light 001 lowshot_mocov3_vit.sh "$task"; done
    for task in 0 1 2; do enqueue light 001 compatibility_finetune_only.sh "$task"; done
    for task in $(seq 0 17); do enqueue light 002 lowshot_cifar100lt.sh "$task"; done
    for task in $(seq 0 5); do enqueue light 003 voc_segmentation_finetune.sh "$task"; done

    # FLUX.1-dev and the exact CIFAR-10-LT TS interaction directly answer
    # reviewer questions and take precedence over broad scaling sweeps.
    for task in $(seq 12 17); do enqueue modern 000 flux_fairness.sh "$task"; done
    for task in $(seq 0 11); do enqueue modern 001 ts_interaction_cifar10lt.sh "$task"; done
    for task in 0 2 13; do enqueue modern 002 compatibility.sh "$task"; done
    printf 'Reprioritized queue for the 48-hour rebuttal critical path.\n'
}

case "${1:-}" in
    init) initialize ;;
    reprioritize-48h) reprioritize_48h ;;
    worker) worker "${2:-}" "${3:-}" ;;
    status) status ;;
    *)
        echo "Usage: $0 {init|reprioritize-48h|status|worker CLASS NAME}" >&2
        exit 2
        ;;
esac
