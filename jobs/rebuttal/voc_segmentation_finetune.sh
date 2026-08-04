#!/bin/bash
#SBATCH --job-name=fomo-voc-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
variant=$([[ "$task" -lt 3 ]] && echo base || echo bridge)
if [[ "$variant" == base ]]; then
  checkpoint="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"
else
  checkpoint="$(find "$CHECKPOINT_ROOT_DIR/rebuttal_factorial_mode_sd3/clane9_imagenet-100/seed_$seed" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
test -n "$checkpoint"
suffix="${FOMO_RUN_SUFFIX:-}"
python paper_work/analysis/voc_segmentation_probe.py \
  --checkpoint "$checkpoint" --root "$BASE_CACHE_DIR/voc2012" \
  --output "$BASE_CACHE_DIR/rebuttal_runs/voc_segmentation_finetune/${variant}_seed${seed}${suffix}.txt" \
  --epochs 20 --seed "$seed" --finetune-encoder
