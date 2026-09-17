#!/bin/bash
#SBATCH --job-name=fomo-resume
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%A_%a.out

# Finish a run that hung part-way instead of repeating cycles it already did.
#
# A cycle here costs the better part of a day, so a run that dies in its last
# cycle has four days of usable work sitting on disk.  Three things persist and
# together make that work recoverable: last.ckpt holds the encoder from the
# final completed cycle, training_progress.json records how many cycles that
# was, and <additional_data_path>_image_counts.pkl records how many repairs
# each earlier cycle contributed, so the dataset rebuilds at its full repaired
# size rather than its original one.  Passing start_cycle skips straight to the
# first unfinished cycle with that state in place.
#
# The resumed cycle draws from a fresh RNG stream, so its repair selection is
# not bit-identical to what the uninterrupted run would have chosen.  Every
# other input to that cycle is the same.
#
# Required: FOMO_RESUME_TAG (run tag), FOMO_RESUME_SEED, FOMO_RESUME_START.
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-0}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-24000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

run_tag="${FOMO_RESUME_TAG:?FOMO_RESUME_TAG is required}"
seed="${FOMO_RESUME_SEED:?FOMO_RESUME_SEED is required}"
start_cycle="${FOMO_RESUME_START:?FOMO_RESUME_START is required}"
ssl_name="${FOMO_RESUME_SSL:-simclr}"
# Must match what this run's earlier cycles used.  The geometry pair scores
# repairs against a panel that includes them ("all"); the other launchers
# leave the default.  Resuming under the other setting would leave a final
# cycle of diagnostics not comparable to the five before it.
diag_ref="${FOMO_RESUME_DIAG_REF:-panel}"
# The repair condition must match the one the earlier cycles ran under.  These
# default to the BRIDGE settings; a baseline arm overrides whichever knob
# defines it, the way its own launcher does.
selection="${FOMO_RESUME_SELECTION:-ood}"
strategy="${FOMO_RESUME_STRATEGY:-mode_window}"
encoder="${FOMO_RESUME_ENCODER:-ssl}"
generator="${FOMO_RESUME_GENERATOR:-stable_diffusion_3}"

run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
checkpoint="$CHECKPOINT_ROOT_DIR/$run_tag/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
test -s "$run_root/generated_image_counts.pkl"

# Refuse to resume past what the run actually completed.  Resuming at a later
# cycle than last.ckpt represents would score an earlier encoder as a finished
# run, which is the exact failure training_progress.json exists to prevent.
completed="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['completed_cycles'])" \
  "$CHECKPOINT_ROOT_DIR/$run_tag/clane9_imagenet-100/seed_${seed}/training_progress.json")"
if [[ "$start_cycle" != "$completed" ]]; then
  echo "start_cycle=$start_cycle but run completed $completed cycles; refusing." >&2
  exit 1
fi

fomo_wait_for_gpu

extra=()
if [[ -n "${FOMO_RESUME_EXTRA:-}" ]]; then read -r -a extra <<<"$FOMO_RESUME_EXTRA"; fi

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl="$ssl_name" \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=6 start_cycle="$start_cycle" \
  n_epochs_per_cycle=100 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy="$strategy" \
  selection_encoder="$encoder" \
  selection_reuse_policy=adaptive repair_once=false \
  ood_distance_metric=normalized_l2 ood_augmentation="${FOMO_RESUME_AUGMENT:-true}" \
  generation_model="$generator" \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  representation_diagnostics_reference="$diag_ref" \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag" "${extra[@]}"
