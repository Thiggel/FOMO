#!/bin/bash
#SBATCH --job-name=fomo-synthetic-provenance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh

# Directly tests recursive synthetic-anchor acquisition versus the safeguard
# that restricts every later selection round to original source examples.
variants=(all_anchors original_only)
task="$SLURM_ARRAY_TASK_ID"; seed="$((task % 3))"; variant="${variants[$((task / 3))]}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
only=false; [[ "$variant" == original_only ]] && only=true
root="$BASE_CACHE_DIR/rebuttal_runs/synthetic_provenance/$variant/seed_${seed}${run_suffix}"; mkdir -p "$root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" max_cycles=3 n_epochs_per_cycle=100 max_steps_per_cycle=3233 \
  train_batch_size=128 val_batch_size=256 num_ood_samples=250 num_generations_per_ood_sample=5 \
  ood_augmentation=true selection_original_only="$only" ood_distance_metric=normalized_l2 \
  generation_model=stable_diffusion_3 additional_data_path="$root/generated" \
  experiment_name="rebuttal_synthetic_provenance_${variant}${run_suffix}"
