#!/bin/bash
#SBATCH --job-name=fomo-flux-fair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-17

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh

# Both FLUX backbones are evaluated at equal NFE.  Redux is the conditioning
# adapter; it is not incorrectly presented as the FLUX dev backbone.
conditions=(
    schnell_steps6 schnell_steps20 schnell_guidance1 schnell_guidance3
    dev_steps6 dev_steps20
)
task="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/last.ckpt"

model_id="black-forest-labs/FLUX.1-schnell"
steps=6
guidance=1
case "$condition" in
    schnell_steps20) steps=20 ;;
    schnell_guidance3) guidance=3 ;;
    dev_steps6) model_id="black-forest-labs/FLUX.1-dev"; guidance=3.5 ;;
    dev_steps20) model_id="black-forest-labs/FLUX.1-dev"; steps=20; guidance=3.5 ;;
esac

run_root="$BASE_CACHE_DIR/rebuttal_runs/flux_fairness/$condition/seed_${seed}${run_suffix}"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr logger=false \
    pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" skip_initial_training=true \
    max_cycles=2 n_epochs_per_cycle=100 max_steps_per_cycle=4850 \
    train_batch_size=128 val_batch_size=256 ood_augmentation=true generation_model=flux \
    flux_model_id="$model_id" flux_num_steps="$steps" flux_guidance="$guidance" flux_batch_size=1 \
    ood_distance_metric=normalized_l2 additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_flux_fair_${condition}${run_suffix}"
