#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1

set -eu

cd "$HOME/FOMO"

. jobs/environment.sh

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${TOTAL_CYCLES:?TOTAL_CYCLES is required}"
: "${EPOCHS_PER_CYCLE:?EPOCHS_PER_CYCLE is required}"
: "${NUM_OOD_SAMPLES:?NUM_OOD_SAMPLES is required}"
: "${START_CYCLE:?START_CYCLE is required}"
: "${STOP_AFTER_CYCLE:?STOP_AFTER_CYCLE is required}"
: "${ADDITIONAL_DATA_PREFIX:?ADDITIONAL_DATA_PREFIX is required}"
: "${DATASET_ID:?DATASET_ID is required}"

mkdir -p job_logs/ablations/cycles

SEED_TOKEN="${SLURM_ARRAY_TASK_ID:-0}"
ADDITIONAL_DATA_PATH="$BASE_CACHE_DIR/${ADDITIONAL_DATA_PREFIX}_seed_${SEED_TOKEN}"

CHECKPOINT_ARG=""
RESUME_ARG=""
if [ "${START_CYCLE}" -gt 0 ]; then
    CHECKPOINT_PATH="${CHECKPOINT_ROOT_DIR}/${EXPERIMENT_NAME}/${DATASET_ID}/seed_${SEED_TOKEN}/last.ckpt"
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "Missing checkpoint for resumed segment: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi
    CHECKPOINT_ARG="checkpoint=${CHECKPOINT_PATH}"
    RESUME_ARG="resume_trainer_state=true"
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles="${TOTAL_CYCLES}" \
    n_epochs_per_cycle="${EPOCHS_PER_CYCLE}" \
    num_ood_samples="${NUM_OOD_SAMPLES}" \
    start_cycle="${START_CYCLE}" \
    stop_after_cycle="${STOP_AFTER_CYCLE}" \
    additional_data_path="${ADDITIONAL_DATA_PATH}" \
    experiment_name="${EXPERIMENT_NAME}" \
    num_runs=3 \
    train_batch_size=512 \
    finetune="${FINETUNE:-false}" \
    ${CHECKPOINT_ARG} \
    ${RESUME_ARG}
