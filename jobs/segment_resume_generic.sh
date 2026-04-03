#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1

set -eu

cd "$HOME/FOMO"

. jobs/environment.sh

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${SSL_NAME:?SSL_NAME is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${TOTAL_CYCLES:?TOTAL_CYCLES is required}"
: "${EPOCHS_PER_CYCLE:?EPOCHS_PER_CYCLE is required}"
: "${START_CYCLE:?START_CYCLE is required}"
: "${STOP_AFTER_CYCLE:?STOP_AFTER_CYCLE is required}"
: "${TRAIN_BATCH_SIZE:?TRAIN_BATCH_SIZE is required}"
: "${ADDITIONAL_DATA_PREFIX:?ADDITIONAL_DATA_PREFIX is required}"
: "${DATASET_ID:?DATASET_ID is required}"

LOG_DIR="${LOG_DIR:-job_logs/segmented}"
mkdir -p "${LOG_DIR}"

SEED_TOKEN="${SLURM_ARRAY_TASK_ID:-0}"
ADDITIONAL_DATA_PATH="$BASE_CACHE_DIR/${ADDITIONAL_DATA_PREFIX}_seed_${SEED_TOKEN}"
CHECKPOINT_DATASET_ID="${CHECKPOINT_DATASET_ID:-${DATASET_ID}}"
CHECKPOINT_PATH="${CHECKPOINT_ROOT_DIR}/${EXPERIMENT_NAME}/${CHECKPOINT_DATASET_ID}/seed_${SEED_TOKEN}/last.ckpt"

declare -a args=(
    "model=${MODEL_NAME}"
    "ssl=${SSL_NAME}"
    "dataset=${DATASET_NAME}"
    "max_cycles=${TOTAL_CYCLES}"
    "n_epochs_per_cycle=${EPOCHS_PER_CYCLE}"
    "start_cycle=${START_CYCLE}"
    "stop_after_cycle=${STOP_AFTER_CYCLE}"
    "additional_data_path=${ADDITIONAL_DATA_PATH}"
    "experiment_name=${EXPERIMENT_NAME}"
    "num_runs=3"
    "train_batch_size=${TRAIN_BATCH_SIZE}"
    "ood_augmentation=${OOD_AUGMENTATION:-true}"
    "finetune=${FINETUNE:-false}"
)

append_arg_if_set() {
    local key="$1"
    local value="${2:-}"
    if [ -n "${value}" ]; then
        args+=("${key}=${value}")
    fi
}

append_arg_if_set "grad_acc_steps" "${GRAD_ACC_STEPS:-}"
append_arg_if_set "num_ood_samples" "${NUM_OOD_SAMPLES:-}"
append_arg_if_set "generation_model" "${GENERATION_MODEL:-}"
append_arg_if_set "flux_batch_size" "${FLUX_BATCH_SIZE:-}"
append_arg_if_set "sd3_batch_size" "${SD3_BATCH_SIZE:-}"
if [ -n "${SPLITS_OVERRIDE:-}" ]; then
    SPLITS_HYDRA_VALUE="${SPLITS_OVERRIDE//|/,}"
    args+=("splits=[${SPLITS_HYDRA_VALUE}]")
fi

if [ "${USE_TEMPERATURE_SCHEDULE:-false}" = "true" ]; then
    args+=("use_temperature_schedule=true")
fi

if [ "${START_CYCLE}" -gt 0 ]; then
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "Missing checkpoint for resumed segment: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi

    # If the segment still performs training, resume trainer state. If it only
    # runs finetuning, load the final weights without trainer-state resume.
    if [ "${START_CYCLE}" -lt "${STOP_AFTER_CYCLE}" ]; then
        args+=("checkpoint=${CHECKPOINT_PATH}" "resume_trainer_state=true")
    else
        args+=("checkpoint=${CHECKPOINT_PATH}")
    fi
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m experiment "${args[@]}"
