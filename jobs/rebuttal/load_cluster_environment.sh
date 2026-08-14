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
        fomo_tmp_gpu="${FOMO_WORKER_NAME:-${SLURM_JOB_ID:-$$}}"
        # /dev/shm is the fast default, but something at node level removes
        # entries from it mid-run on these machines -- the directory is present
        # when python starts, is unique per task, and /dev/shm is nearly empty,
        # yet tempfile calls fail on the missing parent seconds later.  Set
        # FOMO_TMPDIR_ROOT to fall back to ordinary disk when that matters more
        # than speed.
        export TMPDIR="${FOMO_TMPDIR_ROOT:-/dev/shm}/fomo_${USER:-user}_gpu_${fomo_tmp_gpu}"
        export TEMP="$TMPDIR"
        export TMP="$TMPDIR"
        mkdir -p "$TMPDIR"
        ;;
    *)
        echo "Unknown FOMO_CLUSTER=${FOMO_CLUSTER}" >&2
        exit 2
        ;;
esac

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-2}"
export FOMO_ALLOW_DATA_DOWNLOAD=1
export PYTHONUNBUFFERED=1
# Large source archives (notably PASS) must be shared across workers.  TMPDIR
# is deliberately per worker, so using it for dataset downloads would fetch a
# separate multi-GB archive for every GPU.
export FOMO_DATASET_TMPDIR="${FOMO_DATASET_TMPDIR:-$BASE_CACHE_DIR/dataset_downloads}"
mkdir -p "$FOMO_DATASET_TMPDIR"
