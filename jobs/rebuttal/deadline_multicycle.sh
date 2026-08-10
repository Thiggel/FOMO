#!/bin/bash
# Deadline-focused multi-cycle reruns for the ViT, geometry, and TADA rebuttal.
# Each task starts from an already completed source-only checkpoint, then runs
# three matched 60-epoch stages.  BRIDGE/TADA perform one repair before each
# stage; source-only controls receive exactly the same optimization stages.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(
  mocov3_bridge_0 mocov3_base_0
  dino_bridge_0 dino_base_0
  resnet_bridge_0 resnet_tada_0 resnet_base_0
  mocov3_bridge_1 mocov3_base_1
  dino_bridge_1 dino_base_1
  resnet_bridge_1 resnet_tada_1 resnet_base_1
  mocov3_bridge_2 mocov3_base_2
  resnet_bridge_2 resnet_tada_2 resnet_base_2
)

task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition="${conditions[$task]:?Unknown deadline task $task}"
seed="${condition##*_}"
stem="${condition%_*}"
family="${stem%%_*}"
variant="${stem#*_}"

repairs="${FOMO_DEADLINE_REPAIRS:-3}"
epochs="${FOMO_DEADLINE_EPOCHS:-60}"
if (( repairs < 1 )); then
  echo "FOMO_DEADLINE_REPAIRS must be positive" >&2
  exit 2
fi

checkpoint_root="${CHECKPOINT_ROOT_DIR:?CHECKPOINT_ROOT_DIR is required}"
checkpoint=""
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
  resnet)
    model=resnet50
    ssl=simclr
    batch=128
    accum=1
    checkpoint="$checkpoint_root/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
    ;;
  *)
    echo "Unknown family $family" >&2
    exit 2
    ;;
esac

if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
  echo "Missing source checkpoint for $condition: $checkpoint" >&2
  exit 3
fi

augment=false
selection=ood
strategy=mode_window
skip_initial=false
cycles="$repairs"
case "$variant" in
  base)
    ;;
  bridge)
    augment=true
    skip_initial=true
    cycles="$((repairs + 1))"
    ;;
  tada)
    augment=true
    selection=early_loss
    skip_initial=true
    cycles="$((repairs + 1))"
    ;;
  *)
    echo "Unknown variant $variant" >&2
    exit 2
    ;;
esac

run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="deadline_mc${repairs}e${epochs}_${condition}${run_suffix}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

diagnostics=false
if [[ "$family" == resnet ]]; then
  diagnostics=true
fi

python -m experiment \
  dataset=imagenet100_imbalanced model="$model" ssl="$ssl" \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training="$skip_initial" \
  max_cycles="$cycles" n_epochs_per_cycle="$epochs" max_steps_per_cycle=4850 \
  train_batch_size="$batch" grad_acc_steps="$accum" val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy="$strategy" \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle="$diagnostics" \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
