#!/bin/bash
#SBATCH --job-name=fomo-mocov3-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --array=0-5
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
variant=$([[ "$task" -lt 3 ]] && echo base || echo bridge)
checkpoint_dir="$CHECKPOINT_ROOT_DIR/rebuttal_compat_mocov3_vits_${variant}/clane9_imagenet-100/seed_$seed"
checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$checkpoint"
suffix="${FOMO_RUN_SUFFIX:-}"
python -m experiment dataset=imagenet100_imbalanced model=vit_small ssl=moco logger=false \
  pretrain=false finetune=true checkpoint="$checkpoint" finetune_encoder=true \
  finetune_label_fraction=0.10 finetune_seed="$seed" finetune_max_epochs=100 \
  finetune_benchmarks='[CarsFineTune,ImageNet100LTFineTune]' num_runs=1 seed="$seed" \
  experiment_name="rebuttal_lowshot_mocov3_vit_${variant}${suffix}"
