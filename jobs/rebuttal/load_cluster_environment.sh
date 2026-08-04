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
        fomo_tmp_gpu="${CUDA_VISIBLE_DEVICES:-cpu}"
        export TMPDIR="/dev/shm/fomo_${USER:-user}_gpu_${fomo_tmp_gpu}"
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
