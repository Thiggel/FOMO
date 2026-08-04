#!/bin/bash
# Generation-level verification of classifier-free guidance with an empty prompt.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

anchor="$(
  find "$BASE_CACHE_DIR/rebuttal_runs" \
    -path '*/repair_manifests/cycle_0_anchors/anchor_0.png' \
    -type f -print -quit
)"
test -n "$anchor"

suffix="${FOMO_RUN_SUFFIX:-}"
python paper_work/analysis/empty_prompt_guidance_check.py \
  --image "$anchor" \
  --output "$BASE_CACHE_DIR/rebuttal_analysis/empty_prompt_guidance${suffix}"
