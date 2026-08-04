#!/bin/bash
# Clean replacements for repaired ResNet runs whose multiprocessing loaders
# aborted and silently shortened one SSL stage.

set -euo pipefail
export FOMO_NUM_WORKERS=0
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(bridge_0 tada_0 tada_1 tada_2)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition="${conditions[$task]:?Unknown recovery task $task}"
variant="${condition%_*}"
seed="${condition##*_}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  echo "Missing common source checkpoint: $checkpoint" >&2
  exit 3
fi

selection=ood
if [[ "$variant" == tada ]]; then
  selection=early_loss
fi
run_suffix="${FOMO_RUN_SUFFIX:-}"
run_tag="deadline_mc3e60_resnet_${variant}_${seed}${run_suffix}_clean"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"

python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
  max_cycles=4 n_epochs_per_cycle=60 max_steps_per_cycle=4850 \
  train_batch_size=128 grad_acc_steps=1 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection="$selection" ood_selection_strategy=mode_window \
  selection_reuse_policy=adaptive ood_distance_metric=normalized_l2 \
  ood_augmentation=true generation_model=stable_diffusion_3 \
  representation_diagnostics_each_cycle=true \
  additional_data_path="$run_root/generated" experiment_name="$run_tag"
