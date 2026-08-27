#!/bin/sh
set -eu

case "${FOMO_CLUSTER:-}" in
    alex)
        . jobs/clusters/alex_environment.sh
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
        ;;
    lise)
        . jobs/clusters/lise_environment.sh
        ;;
    gruenau|"")
        . jobs/clusters/gruenau_environment.sh
        # Gruenau nodes expose only a small shared /tmp.  Concurrent dataset
        # workers previously filled it and killed otherwise healthy runs.
        #
        # Key the directory on the queue worker when there is one.  Slurm gives
        # every job CUDA_VISIBLE_DEVICES=0 for its single allocated card, so
        # keying on the device made every worker on a node share one temp
        # directory.  Concurrent torch shm sockets then collided there, which
        # kills a run with "torch_shm_manager: could not generate a random
        # directory for manager socket" or a DataLoader worker timeout.
        # Only the queue worker sets FOMO_WORKER_NAME.  Every array launcher
        # fell through to CUDA_VISIBLE_DEVICES, which Slurm sets to 0 for any
        # job holding a single card, so all concurrent tasks on a node shared
        # one directory and deleted it from under each other.  Slurm gives
        # every array task its own job id, so prefer that.
        # Include the array task id explicitly.  SLURM_JOB_ID is not reliably
        # distinct across the tasks of one array here -- task 8 of array 458805
        # reported the same id as task 6 -- so keying on it alone let sibling
        # tasks share a scratch directory and delete it from under each other.
        # That, not any node-level sweeping, is why temp paths kept vanishing:
        # a direct probe showed /dev/shm entries surviving untouched.
        fomo_tmp_slot="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
            fomo_tmp_slot="${fomo_tmp_slot}_${SLURM_ARRAY_TASK_ID}"
        fi
        fomo_tmp_gpu="${FOMO_WORKER_NAME:-$fomo_tmp_slot}"
        # Scratch must not live in /dev/shm here.  systemd-logind runs with the
        # default RemoveIPC=yes, so when any login session of this uid on the
        # node ends -- another job of ours finishing, or even a stray srun --
        # logind wipes every /dev/shm object owned by the uid, including the
        # scratch directories of jobs that are still running.  That, and not
        # any keying collision, is why temp paths kept vanishing mid-run.
        #
        # The site task prolog hands every job SLURM_TMPDIR on node-local ext4
        # (/tmp/$USER/slurm_$SLURM_JOB_ID) and the epilog removes it, so it is
        # both immune to RemoveIPC and self-cleaning.  Only a few kilobytes of
        # tempfiles land here -- datasets go to FOMO_DATASET_TMPDIR -- so the
        # small /tmp partition is ample.  FOMO_TMPDIR_ROOT still overrides.
        fomo_tmp_root="${FOMO_TMPDIR_ROOT:-${SLURM_TMPDIR:-/tmp/${USER:-user}/slurm_${SLURM_JOB_ID:-$$}}}"
        export TMPDIR="${fomo_tmp_root}/fomo_${USER:-user}_gpu_${fomo_tmp_gpu}"
        export TEMP="$TMPDIR"
        export TMP="$TMPDIR"
        mkdir -p "$TMPDIR"
        ;;
    *)
        echo "Unknown FOMO_CLUSTER=${FOMO_CLUSTER}" >&2
        exit 2
        ;;
esac

# Pass shared-tensor handles by file descriptor rather than staging files in
# /dev/shm.  systemd-logind runs with RemoveIPC=yes on these nodes, so it wipes
# every /dev/shm object owned by this uid the moment any login session of ours
# on the node ends -- another array task finishing is enough.  A run whose
# worker tensors live there then dies with "unable to open shared memory
# object", hours in and through no fault of its own.  The file_descriptor
# strategy unlinks immediately and keeps the memory alive through the open
# descriptor, so a sweep has nothing to take.  It was avoided because of
# descriptor limits, which do not bind here: the limit is 524288 and these jobs
# run six workers.
export FOMO_SHARING_STRATEGY="${FOMO_SHARING_STRATEGY:-file_descriptor}"
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-2}"
export FOMO_ALLOW_DATA_DOWNLOAD=1
export PYTHONUNBUFFERED=1
# Large source archives (notably PASS) must be shared across workers.  TMPDIR
# is deliberately per worker, so using it for dataset downloads would fetch a
# separate multi-GB archive for every GPU.
export FOMO_DATASET_TMPDIR="${FOMO_DATASET_TMPDIR:-$BASE_CACHE_DIR/dataset_downloads}"
mkdir -p "$FOMO_DATASET_TMPDIR"
