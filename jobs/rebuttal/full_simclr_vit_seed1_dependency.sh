#!/bin/bash
# Build the missing seed-1 SimCLR ViT-S source checkpoint, then release its
# paired full-schedule tasks into the shared retry queue.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_compat_simclr_vits_base/clane9_imagenet-100/seed_1/last.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  SLURM_ARRAY_TASK_ID=1 \
    FOMO_RUN_GROUP=compatibility \
    FOMO_EXPERIMENT_PREFIX=rebuttal_compat \
    bash jobs/rebuttal/compatibility.sh
fi
test -s "$checkpoint"

queue="${FOMO_RETRY_QUEUE:-$BASE_CACHE_DIR/gruenau_retry_20260725}"
mkdir -p "$queue/modern/pending"
for task in 2 3; do
  target="$queue/modern/pending/00056__full_simclr_vit_5c100__${task}.task"
  exists=false
  for state in pending running completed; do
    if compgen -G "$queue/modern/$state/00056__full_simclr_vit_5c100__${task}*" >/dev/null; then
      exists=true
      break
    fi
  done
  if [[ "$exists" == false ]]; then
    : > "$target"
  fi
done
