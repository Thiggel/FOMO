#!/bin/bash
# Full-schedule ViT follow-up for the reviewer discussion.
# Every branch starts from the same completed 100-epoch source checkpoint and
# receives four additional 100-epoch stages.  Repair branches score the common
# checkpoint before the first additional stage, so both arms receive the same
# number of post-branch optimization stages.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(
  mocov3_base_0 mocov3_bridge_0
  mocov3_base_1 mocov3_bridge_1
  mocov3_base_2 mocov3_bridge_2
  dino_base_0 dino_bridge_0
  dino_base_1 dino_bridge_1
)

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition="${conditions[$task]:?Unknown full ViT task $task}"
seed="${condition##*_}"
stem="${condition%_*}"
family="${stem%%_*}"
variant="${stem#*_}"

checkpoint_root="${CHECKPOINT_ROOT_DIR:?CHECKPOINT_ROOT_DIR is required}"
model=vit_small
ssl=moco
batch=64
accum=2

case "$family" in
  mocov3)
    checkpoint="$checkpoint_root/rebuttal_compat_mocov3_vits_base/clane9_imagenet-100/seed_${seed}/last.ckpt"
    ;;
  dino)
    ssl=dino
    batch=16
    accum=8
    checkpoint="$(
      find "$checkpoint_root" -path \
        "*compat*dino_vits_base*/clane9_imagenet-100/seed_${seed}/last.ckpt" \
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
run_tag="rebuttal_full5e100_${condition}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model="$model" ssl="$ssl" \
  logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training="$skip_initial" \
  max_cycles="$cycles" n_epochs_per_cycle=100 \
  train_batch_size="$batch" grad_acc_steps="$accum" val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=false \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
