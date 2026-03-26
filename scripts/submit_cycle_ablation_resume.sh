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

submit_cycle_chain() {
  local experiment_name="$1"
  local total_cycles="$2"
  local epochs_per_cycle="$3"
  local num_ood_samples="$4"
  local segments_csv="$5"
  local additional_data_prefix="$6"
  local dataset_id="clane9_imagenet-100"

  local previous_job=""
  local start_cycle=0
  local segment_index=1
  IFS=',' read -r -a segment_stops <<< "${segments_csv}"

  for stop_cycle in "${segment_stops[@]}"; do
    local finetune="false"
    if [ "${stop_cycle}" -eq "${total_cycles}" ]; then
      finetune="true"
    fi

    local job_name="${experiment_name}-seg${segment_index}"
    local output_path="job_logs/ablations/cycles/${experiment_name}_seg${segment_index}_%A_%a.out"
    local export_vars="ALL,FOMO_NUM_WORKERS=2,EXPERIMENT_NAME=${experiment_name},TOTAL_CYCLES=${total_cycles},EPOCHS_PER_CYCLE=${epochs_per_cycle},NUM_OOD_SAMPLES=${num_ood_samples},START_CYCLE=${start_cycle},STOP_AFTER_CYCLE=${stop_cycle},FINETUNE=${finetune},ADDITIONAL_DATA_PREFIX=${additional_data_prefix},DATASET_ID=${dataset_id}"

    local jobid
    jobid=$(submit_segment "${previous_job}" \
      --array=0-2 \
      --job-name="${job_name}" \
      --output="${output_path}" \
      --export="${export_vars}" \
      jobs/ablations/cycles/segment_resume.sh)

    echo "${job_name} ${jobid}"

    previous_job="${jobid}"
    start_cycle="${stop_cycle}"
    segment_index=$((segment_index + 1))
  done
}

submit_cycle_chain "ablations_cycles_10" 10 50 250 "5,10" "ablations_cycles_10"
submit_cycle_chain "ablations_cycles_20" 20 25 125 "7,14,20" "ablations_cycles_20"
