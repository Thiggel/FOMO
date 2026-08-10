#!/bin/bash
#SBATCH --job-name=fomo-selector
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-77

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

conditions=(
    metric_raw metric_normalized metric_cosine metric_median
    strategy_top strategy_mode strategy_mode_random strategy_band_random
    strategy_band_fps strategy_all_fps strategy_densest strategy_cluster
    k10 k25 k50 k100 k200
    cutoff95 cutoff97 cutoff99 cutoff995 cutoff100
    alpha1 alpha2 alpha4 alpha8
)
task="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
checkpoint_file="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt"
checkpoint="$(cat "$checkpoint_file")"

metric=normalized_l2
strategy=mode_window
k=100
quantiles='[0.75,0.99]'
alpha=4
case "$condition" in
    metric_raw) metric=raw_l2 ;;
    metric_normalized) ;;
    metric_cosine) metric=cosine ;;
    metric_median) metric=median_normalized ;;
    strategy_top) strategy=top ;;
    strategy_mode) ;;
    strategy_mode_random) strategy=mode_random ;;
    strategy_band_random) strategy=band_random ;;
    strategy_band_fps) strategy=band_fps ;;
    strategy_all_fps) strategy=all_fps ;;
    strategy_densest) strategy=densest_window ;;
    strategy_cluster) strategy=cluster_inverse ;;
    k*) k="${condition#k}" ;;
    cutoff95) quantiles='[0.75,0.95]' ;;
    cutoff97) quantiles='[0.75,0.97]' ;;
    cutoff99) ;;
    cutoff995) quantiles='[0.75,0.995]' ;;
    cutoff100) quantiles='[0.75,1.0]' ;;
    alpha*) alpha="${condition#alpha}" ;;
esac

run_root="$BASE_CACHE_DIR/rebuttal_runs/selector/$condition/seed_${seed}${run_suffix}"
mkdir -p "$run_root"
result="$CHECKPOINT_ROOT_DIR/rebuttal_selector_${condition}${run_suffix}/clane9_imagenet-100/seed_${seed}/result.json"
if fomo_has_full_metric_suite "$result"; then
    echo "Complete result already exists at $result; skipping."
    exit 0
fi
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" \
    skip_initial_training=true max_cycles=2 n_epochs_per_cycle=100 \
    max_steps_per_cycle=4850 train_batch_size=128 val_batch_size=256 \
    k="$k" ood_selection_strategy="$strategy" \
    ood_distance_metric="$metric" \
    ood_mode_histogram_quantile_range="$quantiles" \
    ood_mode_candidate_pool_multiplier="$alpha" \
    ood_augmentation=true generation_model=strong_augmentation \
    additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_selector_${condition}${run_suffix}"
