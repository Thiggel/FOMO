#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

submit_segment() {
  local dependency="$1"
  shift
  local jobid
  if [ -n "${dependency}" ]; then
    jobid=$(sbatch --parsable --dependency="afterok:${dependency}" "$@")
  else
    jobid=$(sbatch --parsable "$@")
  fi
  printf '%s' "${jobid}"
}

submit_chain() {
  local chain_name="$1"
  local output_prefix="$2"
  local log_dir="$3"
  local array_spec="$4"
  local segments="$5"
  shift 5

  local previous_job=""
  local segment_index=1
  local start_cycle=0
  IFS=',' read -r -a stops <<< "${segments}"

  for stop_cycle in "${stops[@]}"; do
    local finetune="false"
    if [ "${start_cycle}" -ge "${stop_cycle}" ]; then
      finetune="true"
    fi

    local job_name="${chain_name}-seg${segment_index}"
    local output_path="${output_prefix}_seg${segment_index}_%A_%a.out"
    local export_vars="ALL,FOMO_NUM_WORKERS=2,LOG_DIR=${log_dir},START_CYCLE=${start_cycle},STOP_AFTER_CYCLE=${stop_cycle},FINETUNE=${finetune}"

    local key
    for key in "$@"; do
      export_vars="${export_vars},${key}"
    done

    local jobid
    jobid=$(submit_segment "${previous_job}" \
      --array="${array_spec}" \
      --job-name="${job_name}" \
      --output="${output_path}" \
      --export="${export_vars}" \
      jobs/segment_resume_generic.sh)

    echo "${job_name} ${jobid}"
    previous_job="${jobid}"
    start_cycle="${stop_cycle}"
    segment_index=$((segment_index + 1))
  done
}

submit_chain \
  "ablations-pretraining-moco" \
  "job_logs/ablations/pretraining/moco" \
  "job_logs/ablations/pretraining" \
  "0" \
  "3,5" \
  "EXPERIMENT_NAME=ablations_pretraining_moco" \
  "MODEL_NAME=resnet50" \
  "SSL_NAME=moco" \
  "DATASET_NAME=imagenet100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=256" \
  "GRAD_ACC_STEPS=2" \
  "NUM_OOD_SAMPLES=500" \
  "SD3_BATCH_SIZE=12" \
  "ADDITIONAL_DATA_PREFIX=ablations_pretraining_moco" \
  "DATASET_ID=clane9_imagenet-100"

submit_chain \
  "ablations-architecture-vit-s" \
  "job_logs/ablations/architecture/vit_s" \
  "job_logs/ablations/architecture" \
  "0-2" \
  "3,5" \
  "EXPERIMENT_NAME=ablations_architecture_vit-s" \
  "MODEL_NAME=vit_small" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=imagenet100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "NUM_OOD_SAMPLES=500" \
  "ADDITIONAL_DATA_PREFIX=ablations_architecture_vit-s" \
  "DATASET_ID=clane9_imagenet-100"

submit_chain \
  "ablations-architecture-vit-b" \
  "job_logs/ablations/architecture/vit_b" \
  "job_logs/ablations/architecture" \
  "0-2" \
  "3,5" \
  "EXPERIMENT_NAME=ablations_architecture_vit-b" \
  "MODEL_NAME=vit_base" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=imagenet100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "NUM_OOD_SAMPLES=500" \
  "ADDITIONAL_DATA_PREFIX=ablations_architecture_vit-b" \
  "DATASET_ID=clane9_imagenet-100"

submit_chain \
  "ablations-generation-flux" \
  "job_logs/ablations/generation/flux" \
  "job_logs/ablations/generation" \
  "0-2" \
  "2,4,5" \
  "EXPERIMENT_NAME=ablations_generation_flux" \
  "MODEL_NAME=resnet50" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=imagenet100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "NUM_OOD_SAMPLES=500" \
  "GENERATION_MODEL=flux" \
  "FLUX_BATCH_SIZE=2" \
  "ADDITIONAL_DATA_PREFIX=ablations_generation_flux" \
  "DATASET_ID=clane9_imagenet-100"

# SOTA reruns only need the full pretraining segment and a finetune-only segment.
submit_chain \
  "sota-cifar-100-lt-bridge" \
  "job_logs/sota/cifar-100-lt/bridge" \
  "job_logs/sota/cifar-100-lt" \
  "2" \
  "5,5" \
  "EXPERIMENT_NAME=sota_cifar-100-lt_bridge" \
  "MODEL_NAME=resnet50" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=cifar100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "ADDITIONAL_DATA_PREFIX=sota_cifar-100-lt_bridge" \
  "DATASET_ID=uoft-cs_cifar100"

submit_chain \
  "sota-cifar-100-lt-bridge-ts" \
  "job_logs/sota/cifar-100-lt/bridge_ts" \
  "job_logs/sota/cifar-100-lt" \
  "1" \
  "5,5" \
  "EXPERIMENT_NAME=sota_cifar-100-lt_bridge-ts" \
  "MODEL_NAME=resnet50" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=cifar100_imbalanced" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "USE_TEMPERATURE_SCHEDULE=true" \
  "ADDITIONAL_DATA_PREFIX=sota_cifar-100-lt_bridge-ts" \
  "DATASET_ID=uoft-cs_cifar100"

submit_chain \
  "sota-pass-subset-bridge-ts" \
  "job_logs/sota/pass-subset/bridge_ts" \
  "job_logs/sota/pass-subset" \
  "0-1" \
  "5,5" \
  "EXPERIMENT_NAME=sota_pass-subset_bridge-ts" \
  "MODEL_NAME=resnet50" \
  "SSL_NAME=simclr" \
  "DATASET_NAME=pass_subset" \
  "TOTAL_CYCLES=5" \
  "EPOCHS_PER_CYCLE=100" \
  "TRAIN_BATCH_SIZE=512" \
  "USE_TEMPERATURE_SCHEDULE=true" \
  "SD3_BATCH_SIZE=12" \
  "SPLITS_OVERRIDE=1.0|0.0|0.0" \
  "ADDITIONAL_DATA_PREFIX=sota_pass-subset_bridge-ts" \
  "DATASET_ID=pass_subset"
