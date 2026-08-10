#!/bin/bash
# Same sparse anchors and BLIP captions as the from-noise control, retaining
# the anchor image as the SDEdit initialization to isolate image conditioning.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
run_tag="iclr_captioned_img2img_${seed}${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"
python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmark_suite=paper_full \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=4 n_epochs_per_cycle=60 \
  max_steps_per_cycle=4850 train_batch_size=128 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood selection_encoder=ssl \
  ood_selection_strategy=mode_window ood_distance_metric=normalized_l2 \
  selection_reuse_policy=adaptive ood_augmentation=true \
  generation_model=stable_diffusion_3_captioned_img2img \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" experiment_name="$run_tag"
