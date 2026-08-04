#!/bin/bash
#SBATCH --job-name=fomo-toy-support
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
python paper_work/analysis/toy_recoverable_support.py --output "$BASE_CACHE_DIR/rebuttal_analysis/toy_recoverable_support"
