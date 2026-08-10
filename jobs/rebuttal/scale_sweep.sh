#!/bin/bash
#SBATCH --job-name=fomo-scale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-47
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
sizes=(10000 25000 50000 100000); budgets=(fixed fraction); variants=(base bridge)
task="$SLURM_ARRAY_TASK_ID"; seed="$((task % 3))"; cell="$((task/3))"; size="${sizes[$((cell/4))]}"; budget="${budgets[$(((cell/2)%2))]}"; variant="${variants[$((cell%2))]}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
anchors=500; [[ "$budget" == fraction ]] && anchors="$((size/20))"
augment=false; cycles=1; steps=9700; [[ "$variant" == bridge ]] && { augment=true; cycles=2; steps=4850; }
root="$BASE_CACHE_DIR/rebuttal_runs/scale_${size}_${budget}_${variant}/seed_${seed}${run_suffix}"; mkdir -p "$root"
result="$CHECKPOINT_ROOT_DIR/rebuttal_scale_${size}_${budget}_${variant}${run_suffix}/experiment_dataset_hf_scripts_pass_subset.py/seed_${seed}/result.json"
if [[ -s "$result" ]] && jq -e '(.cars_test_accuracy|numbers) and (.aircraft_test_accuracy|numbers) and (.flowers_test_accuracy|numbers) and (.imagenet100lt_test_accuracy|numbers)' "$result" >/dev/null; then
  echo "Complete result already exists at $result; skipping."
  exit 0
fi
python -m experiment dataset=pass_subset "dataset.split='train[:${size}]'" model=resnet50 ssl=simclr logger=false pretrain=true finetune=true \
 finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,ImageNet100LTFineTune]' num_runs=1 seed="$seed" max_cycles="$cycles" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" train_batch_size=128 val_batch_size=256 num_ood_samples="$anchors" ood_augmentation="$augment" generation_model=strong_augmentation ood_distance_metric=normalized_l2 additional_data_path="$root/generated" experiment_name="rebuttal_scale_${size}_${budget}_${variant}${run_suffix}"
