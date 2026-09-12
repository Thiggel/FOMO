#!/bin/bash
#SBATCH --job-name=fomo-geompair
#SBATCH --partition=gpu,gpu-staff
#SBATCH --gres=gpu:1
# The reference set here holds the generated repairs as well as the panel, so
# the diagnostics pass embeds and indexes more than the other launchers do and
# does not fit a 24 GiB card beside a co-tenant: seed 1 of the no-repair arm
# reached 10.4 GiB and died next to a neighbour holding 11.5 GiB of 21.98 GiB.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#
# The paired run behind the geometry figure: repair against no repair, three
# seeds, on the protocol the paper reports.
#
# The existing per-cycle diagnostics cannot answer the question the figure is
# meant to answer.  They measure a fixed panel of original images against an
# index built from that same panel, so the generated repairs are not candidate
# neighbours and a repair cannot reduce the radius of the anchor it was made
# for.  Measured that way the repaired and unrepaired arms differ by nothing
# consistent: the inequality of local support improves for DINO on both seeds
# and worsens for MAE on both.
#
# representation_diagnostics_reference=all keeps the measured population fixed,
# so cycles stay comparable, and lets the repairs into the reference set.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-6}"
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-16000}"
export FOMO_GPU_WAIT_ATTEMPTS="${FOMO_GPU_WAIT_ATTEMPTS:-240}"

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
if (( task < 3 )); then condition=norepair; augment=false; else condition=bridge; augment=true; fi

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_geompair5c100_${condition}_${seed}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=6 \
  n_epochs_per_cycle=100 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive repair_once=false \
  ood_distance_metric=normalized_l2 ood_augmentation="$augment" \
  generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  representation_diagnostics_reference=all \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
