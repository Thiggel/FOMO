#!/bin/bash
# Full paired SimCLR and ViT-S runs for the objective by backbone table.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

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

python -m experiment \
  dataset=imagenet100_imbalanced model=vit_small ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training="$skip_initial" \
  max_cycles="$cycles" n_epochs_per_cycle=100 \
  train_batch_size=64 grad_acc_steps=2 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=false \
  additional_data_path="$run_root/generated" \
  experiment_name="$run_tag"
