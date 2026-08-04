#!/bin/bash
#SBATCH --job-name=fomo-geom-acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-2
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
seed="${SLURM_ARRAY_TASK_ID}"
base="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"
bridge="$(find "$CHECKPOINT_ROOT_DIR/rebuttal_factorial_mode_sd3/clane9_imagenet-100/seed_$seed" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$base" && test -n "$bridge"
suffix="${FOMO_RUN_SUFFIX:-}"
python paper_work/analysis/geometry_accuracy_correlation.py \
  --base "$base" --bridge "$bridge" --seed "$seed" \
  --output "$BASE_CACHE_DIR/rebuttal_analysis/geometry_accuracy/seed_${seed}${suffix}.json"
