#!/bin/bash
#SBATCH --job-name=fomo-compat-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-2
set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Complete downstream evaluation from already-finished ViT pretraining
# checkpoints instead of wasting the rebuttal window on repeated pretraining.
conditions=(simclr_vits_bridge dino_vits_base dino_vits_bridge)
seeds=(0 0 1)
task="${SLURM_ARRAY_TASK_ID}"
condition="${conditions[$task]}"
seed="${seeds[$task]}"
model=vit_small
ssl=simclr
[[ "$condition" == dino_* ]] && ssl=dino

checkpoint=""
for root in "$CHECKPOINT_ROOT_DIR"/rebuttal_compat*_"$condition"; do
  checkpoint_dir="$root/clane9_imagenet-100/seed_$seed"
  [[ -d "$checkpoint_dir" ]] || continue
  candidate="$(find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -n "$candidate" ]]; then checkpoint="$candidate"; break; fi
done
test -n "$checkpoint"
suffix="${FOMO_RUN_SUFFIX:-}"
python -m experiment dataset=imagenet100_imbalanced model="$model" ssl="$ssl" logger=false \
  pretrain=false finetune=true checkpoint="$checkpoint" \
  finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,PetsFineTune,CIFAR10FineTuner,CIFAR100FineTuner,ImageNet100LTFineTune]' \
  num_runs=1 seed="$seed" \
  experiment_name="rebuttal_compat_finetune_${condition}${suffix}"
