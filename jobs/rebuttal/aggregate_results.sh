#!/bin/bash
#SBATCH --job-name=fomo-aggregate-results
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
python paper_work/analysis/aggregate_rebuttal_results.py \
  --root "$CHECKPOINT_ROOT_DIR" --output "$BASE_CACHE_DIR/rebuttal_analysis/results_ledger"
