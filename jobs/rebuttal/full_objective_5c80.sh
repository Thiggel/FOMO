#!/bin/bash
#SBATCH --job-name=fomo-obj5c80
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
# 24GB cards cannot hold the multi-crop batch alongside an MPS co-tenant.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00
#SBATCH --array=0-11
#
# Full-schedule MAE and DINOv2 comparison, five cycles at 80 epochs.
#
# These two objectives were previously only run under jobs/rebuttal/
# compatibility.sh, which gives the base arm one cycle and the repair arm two.
# That is a single repair stage, and the percentile sweep shows that no
# acquisition rule separates from any other after one stage: every score band
# from the densest quartile to the extreme tail lands within 1.32 points.  A
# compatibility verdict taken from that protocol is therefore uninformative in
# either direction, which is why both objectives are rerun here on the same
# five-cycle schedule as the SimCLR and MoCo v3 arms.
#
# 80 epochs per cycle, not 100: DINOv2 carries multi-crop and a patch-level
# objective at batch 16 with 8 accumulation steps, and 5 x 100 epochs does not
# finish inside the 6-day partition limit.  Both objectives use 80 so the
# schedule is shared.
#
# Base and BRIDGE branch from one common per-seed source checkpoint and receive
# the same number of post-branch optimizer stages.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(
  mae_base_0 mae_bridge_0
  mae_base_1 mae_bridge_1
  mae_base_2 mae_bridge_2
  dinov2_base_0 dinov2_bridge_0
  dinov2_base_1 dinov2_bridge_1
  dinov2_base_2 dinov2_bridge_2
)

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition="${conditions[$task]:?Unknown objective task $task}"
seed="${condition##*_}"
stem="${condition%_*}"
family="${stem%%_*}"
variant="${stem#*_}"

checkpoint_root="${CHECKPOINT_ROOT_DIR:?CHECKPOINT_ROOT_DIR is required}"
model=vit_small

case "$family" in
  mae)
    ssl=mae
    batch=64
    accum=2
    checkpoint="$checkpoint_root/rebuttal_compat_mae_vits_base/clane9_imagenet-100/seed_${seed}/last.ckpt"
    ;;
  dinov2)
    ssl=dinov2
    batch=16
    accum=8
    checkpoint="$(
      find "$checkpoint_root" -path \
        "*source_dinov2_vits*/clane9_imagenet-100/seed_${seed}/last.ckpt" \
        -type f -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
    ;;
  *)
    echo "Unknown family $family" >&2
    exit 2
    ;;
esac

if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
  echo "Missing common source checkpoint for $condition: $checkpoint" >&2
  echo "For dinov2, run jobs/rebuttal/dinov2_source.sh first." >&2
  exit 3
fi

augment=false
skip_initial=false
cycles=4
if [[ "$variant" == bridge ]]; then
  augment=true
  skip_initial=true
  cycles=5
fi

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="rebuttal_full5e80_${condition}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model="$model" ssl="$ssl" \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training="$skip_initial" \
  max_cycles="$cycles" n_epochs_per_cycle=80 \
  train_batch_size="$batch" grad_acc_steps="$accum" val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=false \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
