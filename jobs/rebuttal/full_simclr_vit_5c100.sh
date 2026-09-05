#!/bin/bash
#SBATCH --job-name=fomo-simclr5c100
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
# SimCLR bridge measured 28 hours.
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
# Full paired SimCLR and ViT-S runs for the objective by backbone table.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

# Submitted directly through sbatch now that the retry queue worker is gone, so
# the hardening the other launchers carry has to live here too.
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-6}"
# Persistent workers stay off.  Keeping the pool alive across a cycle boundary
# races with the teardown of the previous cycle's combined loader and aborts the
# run with "terminate called without an active exception".
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-18000}"
# Wait out a co-tenant rather than hand back a slot a multi-day run needs.
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

conditions=(
  base_0 bridge_0
  base_1 bridge_1
  base_2 bridge_2
)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition="${conditions[$task]:?Unknown SimCLR ViT task $task}"
seed="${condition##*_}"
variant="${condition%_*}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_compat_simclr_vits_base/clane9_imagenet-100/seed_${seed}/last.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  checkpoint="$(
    find "$CHECKPOINT_ROOT_DIR" -path \
      "*compat*simclr_vits_base*/clane9_imagenet-100/seed_${seed}/last.ckpt" \
      -type f -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
fi
test -n "$checkpoint"
test -s "$checkpoint"

augment=false
skip_initial=false
cycles=4
if [[ "$variant" == bridge ]]; then
  augment=true
  skip_initial=true
  cycles=5
fi

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_full5e100_simclr_vits_${condition}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
  dataset=imagenet100_imbalanced model=vit_small ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training="$skip_initial" \
  max_cycles="$cycles" n_epochs_per_cycle=100 \
  train_batch_size=64 grad_acc_steps=2 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
