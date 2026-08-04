#!/bin/bash
#SBATCH --job-name=fomo-voc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"; . jobs/rebuttal/load_cluster_environment.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
seed="$((SLURM_ARRAY_TASK_ID % 3))"; variant=$([[ $SLURM_ARRAY_TASK_ID -lt 3 ]] && echo base || echo bridge)
if [[ "$variant" == base ]]; then ckpt="$(cat "$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt")"; else ckpt="$(find "$CHECKPOINT_ROOT_DIR/rebuttal_factorial_mode_sd3/clane9_imagenet-100/seed_$seed" -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"; fi
python paper_work/analysis/voc_segmentation_probe.py --checkpoint "$ckpt" --root "$BASE_CACHE_DIR/voc2012" --output "$BASE_CACHE_DIR/rebuttal_runs/voc_segmentation/${variant}_seed${seed}.txt"
