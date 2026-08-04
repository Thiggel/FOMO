#!/bin/bash
#SBATCH --job-name=fomo-dinov3-teacher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# External-data control: DINOv3 alignment on existing real source images,
# without any generated samples, and the corresponding BRIDGE+teacher arm.
conditions=(teacher_only teacher_bridge)
task="${SLURM_ARRAY_TASK_ID}"; seed="$((task % 3))"; condition="${conditions[$((task / 3))]}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
augment=false; cycles=1; steps=9700
if [[ "$condition" == teacher_bridge ]]; then augment=true; cycles=2; steps=4850; fi
run_root="$BASE_CACHE_DIR/rebuttal_runs/dinov3_teacher/$condition/seed_${seed}${run_suffix}"; mkdir -p "$run_root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
  train_batch_size=128 val_batch_size=256 ood_augmentation="$augment" generation_model=stable_diffusion_3 \
  external_teacher_model=facebook/dinov3-vits16-pretrain-lvd1689m external_teacher_weight=0.1 \
  ood_distance_metric=normalized_l2 additional_data_path="$run_root/generated" experiment_name="rebuttal_dinov3_${condition}${run_suffix}"
