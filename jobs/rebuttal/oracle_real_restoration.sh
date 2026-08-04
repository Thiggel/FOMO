#!/bin/bash
#SBATCH --job-name=fomo-oracle-real
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$SLURM_ARRAY_TASK_ID"; checkpoint="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"; root="$BASE_CACHE_DIR/rebuttal_runs/oracle_real/seed_${seed}${run_suffix}"; mkdir -p "$root"
python -m experiment dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
 finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true max_cycles=2 n_epochs_per_cycle=100 max_steps_per_cycle=4850 train_batch_size=128 val_batch_size=256 sample_selection=oracle_real ood_augmentation=true remove_diffusion=true num_ood_samples=500 num_generations_per_ood_sample=5 ood_distance_metric=normalized_l2 additional_data_path="$root/generated" experiment_name="rebuttal_oracle_real${run_suffix}"
