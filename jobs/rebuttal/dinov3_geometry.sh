#!/bin/bash
#SBATCH --job-name=fomo-geometry
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
run_suffix="${FOMO_RUN_SUFFIX:-}"
python paper_work/analysis/dinov3_geometry_report.py --inputs "$BASE_CACHE_DIR/rebuttal_runs/causal_factorial/mode_sd3/seed_*/generated/**/*.h5" "$BASE_CACHE_DIR/rebuttal_runs/causal_factorial/uniform_sd3/seed_*/generated/**/*.h5" "$BASE_CACHE_DIR/rebuttal_runs/causal_factorial/top_sd3/seed_*/generated/**/*.h5" --output "$BASE_CACHE_DIR/rebuttal_analysis/dinov3_geometry${run_suffix}"
