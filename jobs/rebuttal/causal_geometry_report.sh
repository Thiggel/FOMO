#!/bin/bash
#SBATCH --job-name=fomo-geometry-report
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
python paper_work/analysis/causal_geometry_report.py \
  --inputs "$BASE_CACHE_DIR/rebuttal_runs/causal_geometry/**/representation_diagnostics/cycle_*.json" \
  --output "$BASE_CACHE_DIR/rebuttal_analysis/causal_geometry"
