#!/bin/bash

# Wait until the allocated GPU actually has room before starting a run.
#
# These nodes run MPS and other tenants attach to the same cards outside
# Slurm's accounting, so an allocation regularly comes with only a couple of
# GB free.  A run that starts anyway dies with a CUDA OOM part-way through and
# loses everything it had computed.  Waiting costs nothing -- the allocation is
# already ours -- and a job that never gets room exits before doing work, so it
# can simply be resubmitted.
fomo_wait_for_gpu() {
  local required="${FOMO_MIN_FREE_GPU_MIB:-12000}"
  local attempts="${FOMO_GPU_WAIT_ATTEMPTS:-60}"
  local delay="${FOMO_GPU_WAIT_SECONDS:-60}"

  # Assert the scratch directory immediately before the run starts.  Torch's
  # shm manager creates its socket directory under TMPDIR and fails with
  # "could not generate a random directory for manager socket" if the path is
  # missing, which has repeatedly killed jobs seconds after launch.  Something
  # on these nodes removes /dev/shm entries between job setup and execution,
  # so recreating it here rather than trusting setup is the reliable fix.
  [[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

  command -v nvidia-smi >/dev/null 2>&1 || return 0

  # Read the card this job was actually given.  ``head -n 1`` reported card 0
  # of the node no matter which one Slurm allocated, so a task holding a busy
  # card three slots along saw a neighbour's free memory and walked straight
  # into a CUDA OOM, while a task holding an empty card could sit in the wait
  # loop until it gave up.  Slurm's cgroup usually narrows the listing to the
  # allocated device, in which case the index is renumbered to 0 and the
  # lookup below falls through to the single row; where it does not, the
  # CUDA_VISIBLE_DEVICES index selects the right row.
  local attempt free query visible
  visible="${CUDA_VISIBLE_DEVICES%%,*}"
  for (( attempt = 1; attempt <= attempts; attempt++ )); do
    query="$(nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits 2>/dev/null)"
    free=""
    if [[ -n "$visible" ]]; then
      free="$(awk -F', *' -v i="$visible" '$1 == i { print $2 }' <<<"$query" |
        head -n 1 | tr -d '[:space:]')"
    fi
    if [[ -z "$free" ]]; then
      free="$(awk -F', *' 'NR == 1 { print $2 }' <<<"$query" | tr -d '[:space:]')"
    fi
    [[ "$free" =~ ^[0-9]+$ ]] || return 0
    if (( free >= required )); then
      return 0
    fi
    echo "Waiting for GPU memory: ${free} MiB free, need ${required} MiB" \
         "(attempt ${attempt}/${attempts})"
    sleep "$delay"
  done

  echo "Giving up: GPU never had ${required} MiB free. Resubmit this task." >&2
  return 75
}

# Canonical result contract for paper-facing experiments. Keep this in sync
# with FinetuningBenchmarks.benchmark_suites["paper_full"].
fomo_has_full_metric_suite() {
  local result_file="$1"
  [[ -s "$result_file" ]] && jq -e '
    [
      .cars_test_accuracy,
      .carsknn_knn_test_accuracy,
      .aircraft_test_accuracy,
      .aircraftknn_knn_test_accuracy,
      .flowers_test_accuracy,
      .flowersknn_knn_test_accuracy,
      .pets_test_accuracy,
      .petsknn_knn_test_accuracy,
      .cifar10r_test_accuracy,
      .cifar10knn_knn_test_accuracy,
      .cifar100r_test_accuracy,
      .cifar100knn_knn_test_accuracy,
      .imagenet100lt_test_accuracy,
      .imagenet100ltknn_knn_test_accuracy
    ] | all(.[]; type == "number")
  ' "$result_file" >/dev/null
}
