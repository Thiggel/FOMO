#!/bin/bash
# Label-free AIDE-style control: a frozen VLM identifies rare visual concepts,
# a captioner verbalizes selected anchors, and SD3 generates from text/noise.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
seed="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_${seed}/last.ckpt"
test -s "$checkpoint"
run_tag="iclr_aide_ssl_clip_cluster_${seed}${FOMO_RUN_SUFFIX:-}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_tag"
mkdir -p "$run_root"
python -m experiment \
  dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
  logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  skip_initial_training=true max_cycles=4 n_epochs_per_cycle=60 \
  max_steps_per_cycle=4850 train_batch_size=128 val_batch_size=256 \
  num_ood_samples=500 num_generations_per_ood_sample=5 \
  sample_selection=ood selection_encoder=clip \
  ood_selection_strategy=cluster_inverse ood_distance_metric=normalized_l2 \
  selection_reuse_policy=adaptive ood_augmentation=true \
  generation_model=stable_diffusion_3_t2i \
  representation_diagnostics_each_cycle=true \
  representation_diagnostics_save_samples=true \
  additional_data_path="$run_root/generated" experiment_name="$run_tag"
