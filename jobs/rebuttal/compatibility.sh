#!/bin/bash
#SBATCH --job-name=fomo-compat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-29

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

# Healthy same-source pairs.  Every adjacent base/BRIDGE pair shares the same
# objective, backbone, exposure budget, and seed.
conditions=(
    simclr_vits_base simclr_vits_bridge
    dino_r50_base dino_r50_bridge
    dino_vits_base dino_vits_bridge
    mocov3_vits_base mocov3_vits_bridge
    mae_vits_base mae_vits_bridge
)
task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"

model=vit_small
ssl=simclr
batch=64
accum=2
case "$condition" in
    dino_r50_*) model=resnet50; ssl=dino; batch=16; accum=8 ;;
    dino_vits_*) ssl=dino; batch=16; accum=8 ;;
    mocov3_vits_*) ssl=moco ;;
    mae_vits_*) ssl=mae ;;
esac
if [[ "$condition" == *_bridge ]]; then
    cycles=2
    augment=true
else
    cycles=1
    augment=false
fi

run_group="${FOMO_RUN_GROUP:-compatibility}"
experiment_prefix="${FOMO_EXPERIMENT_PREFIX:-rebuttal_compat}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_group/$condition/seed_$seed"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model="$model" ssl="$ssl" \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" \
    train_batch_size="$batch" grad_acc_steps="$accum" val_batch_size=256 \
    max_cycles="$cycles" n_epochs_per_cycle="$((100 / cycles))" \
    max_steps_per_cycle="$((9700 / cycles))" \
    ood_augmentation="$augment" ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="${experiment_prefix}_${condition}"
