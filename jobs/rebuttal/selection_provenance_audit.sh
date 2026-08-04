#!/bin/bash
#SBATCH --job-name=fomo-selection-audit
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
python paper_work/analysis/selection_provenance_audit.py \
  --inputs "$BASE_CACHE_DIR/rebuttal_runs/**/ood_diagnostics/*/selection.npz" \
  --output "$BASE_CACHE_DIR/rebuttal_analysis/selection_provenance"
