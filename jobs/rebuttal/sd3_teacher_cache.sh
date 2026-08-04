#!/bin/bash
#SBATCH --job-name=fomo-sd3-teacher-cache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-2
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python paper_work/analysis/cache_sd3_vae_teacher.py --seed "$SLURM_ARRAY_TASK_ID" \
  --output "$BASE_CACHE_DIR/sd3_teacher_cache/seed_${SLURM_ARRAY_TASK_ID}.pt"
