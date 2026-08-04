#!/bin/bash
#SBATCH --job-name=fomo-repair-fidelity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
run_suffix="${FOMO_RUN_SUFFIX:-}"
python paper_work/analysis/repair_fidelity_report.py \
  --manifests "$BASE_CACHE_DIR/rebuttal_runs/**/repair_manifests/cycle_*.json" \
  --output "$BASE_CACHE_DIR/rebuttal_analysis/repair_fidelity${run_suffix}"
