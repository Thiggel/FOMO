#!/bin/bash
#SBATCH --job-name=fomo-generator
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-41

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

# Equal-NFE points plus label-free SD3 strength/guidance screens. FLUX Redux
# has no img2img strength parameter, so strength is not falsely presented as a
# matched knob for that architecture.
conditions=(
    sd3_steps6 sd3_steps20 flux_steps6 flux_steps20
    sd3_strength03 sd3_strength05 sd3_strength06
    sd3_strength07 sd3_strength09
    sd3_guidance1 sd3_guidance3 sd3_guidance5 sd3_guidance75
    flux_guidance1
)
task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
checkpoint_file="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt"
checkpoint="$(cat "$checkpoint_file")"

generator=stable_diffusion_3
sd3_steps=20
sd3_strength=0.6
sd3_guidance=5
flux_steps=6
flux_guidance=1
case "$condition" in
    sd3_steps6) sd3_steps=6 ;;
    sd3_steps20) ;;
    flux_steps6) generator=flux ;;
    flux_steps20) generator=flux; flux_steps=20 ;;
    sd3_strength03) sd3_strength=0.3 ;;
    sd3_strength05) sd3_strength=0.5 ;;
    sd3_strength06) ;;
    sd3_strength07) sd3_strength=0.7 ;;
    sd3_strength09) sd3_strength=0.9 ;;
    sd3_guidance1) sd3_guidance=1 ;;
    sd3_guidance3) sd3_guidance=3 ;;
    sd3_guidance5) ;;
    sd3_guidance75) sd3_guidance=7.5 ;;
    flux_guidance1) generator=flux ;;
esac

run_group="${FOMO_RUN_GROUP:-generator}"
experiment_prefix="${FOMO_EXPERIMENT_PREFIX:-rebuttal_generator}"
run_root="$BASE_CACHE_DIR/rebuttal_runs/$run_group/$condition/seed_$seed"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" \
    skip_initial_training=true max_cycles=2 n_epochs_per_cycle=100 \
    max_steps_per_cycle=4850 train_batch_size=128 val_batch_size=256 \
    ood_augmentation=true generation_model="$generator" \
    sd3_num_steps="$sd3_steps" sd3_strength="$sd3_strength" \
    sd3_guidance="$sd3_guidance" flux_num_steps="$flux_steps" \
    flux_guidance="$flux_guidance" flux_batch_size=1 ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="${experiment_prefix}_${condition}"
