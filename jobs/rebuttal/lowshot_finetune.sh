#!/bin/bash
#SBATCH --job-name=fomo-lowshot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-17

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

variants=(base bridge)
fractions=(0.01 0.10 1.0)
task="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$((task % 3))"
cell="$((task / 3))"
fraction="${fractions[$((cell % 3))]}"
variant="${variants[$((cell / 3))]}"

if [ "$variant" = base ]; then
    checkpoint_dir="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed"
else
    checkpoint_dir="$CHECKPOINT_ROOT_DIR/rebuttal_factorial_mode_sd3/clane9_imagenet-100/seed_$seed"
fi
checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$checkpoint"

python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=false finetune=true checkpoint="$checkpoint" \
    finetune_encoder=true finetune_label_fraction="$fraction" \
    finetune_seed="$seed" finetune_max_epochs=100 \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" \
    experiment_name="rebuttal_lowshot_${variant}_${fraction}${run_suffix}"
