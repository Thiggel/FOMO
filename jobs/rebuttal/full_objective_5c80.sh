#!/bin/bash
#SBATCH --job-name=fomo-obj5c80
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
# 24GB cards cannot hold the multi-crop batch alongside an MPS co-tenant.
#SBATCH --exclude=gruenau1,gruenau2
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
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
. jobs/rebuttal/full_metric_suite.sh

# The cluster default of 2 dataloader workers starves the GPU: multi-crop
# augmentation decodes and transforms ten views per sample on the CPU, and a
# survey of our running jobs found most GPUs at 0 percent utilisation with
# memory resident.  Persistent workers matter as much as the count, because an
# epoch here is only a few hundred steps and the pool was otherwise being torn
# down and rebuilt every epoch.
export FOMO_NUM_WORKERS="${FOMO_NUM_WORKERS:-6}"
# Persistent workers are off again.  Keeping the pool alive across a cycle
# boundary races with the teardown of the previous cycle's combined loader:
# the run prints "terminate called without an active exception" as the stage
# ends and the worker dies of SIGABRT.  Seven runs were lost to this in one
# morning, every one of them at a cycle boundary and none inside a stage.  The
# throughput this was meant to buy comes mostly from the worker count and the
# prefetch depth below, which are unaffected.
export FOMO_PERSISTENT_WORKERS="${FOMO_PERSISTENT_WORKERS:-0}"
export FOMO_PREFETCH_FACTOR="${FOMO_PREFETCH_FACTOR:-2}"
# DINOv2 seeds 1 and 2 died of CUDA OOM beside a co-tenant that held 42 of the
# card's 47 GiB.  The failing request was 20 MiB, so the card was not full, it
# was fragmented: multi-crop sends ten views of differing spatial size through
# the backbone every step, and the caching allocator cannot reuse a block sized
# for a 224 crop to serve a 96 one.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FOMO_MIN_FREE_GPU_MIB="${FOMO_MIN_FREE_GPU_MIB:-40000}"


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
    checkpoint="$(
      find "$checkpoint_root" -path \
        "*source_mae_vits*/clane9_imagenet-100/seed_${seed}/last.ckpt" \
        -type f -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
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
  echo "Run jobs/rebuttal/objective_source.sh first." >&2
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

fomo_wait_for_gpu

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
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
