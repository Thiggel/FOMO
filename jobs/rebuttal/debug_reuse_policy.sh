#!/bin/bash
#SBATCH --job-name=fomo-dbgreuse
#SBATCH --partition=wbimlgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/vol/home-vol2/ml/laitenbf/FOMO_runtime/logs/full_metric_queue/dbgreuse_%A_%a.out
#
# Minimal reproduction of the reuse-policy defect.
#
# selection_reuse_policy=static_first_cycle selects demonstrably different
# anchors from adaptive -- the repair manifests overlap on only 14 of 500 from
# cycle 1 onward -- yet both produce bit-identical encoders.  The divergence is
# lost somewhere between anchor selection and the trained weights.
#
# This runs the same comparison at a scale that finishes in minutes: three
# cycles, one epoch each, twenty anchors, conventional augmentation so no
# diffusion model is loaded, and no downstream benchmarks.  Diff the two
# resulting last.ckpt files; if they match at this scale the defect reproduces
# and can be bisected quickly.

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

# The small configuration below reproduces *correct* behaviour: the two
# policies diverge as they should (max weight difference 2.9e-01).  The real
# runs that come out bit-identical differ from it in scale, schedule and, most
# suspiciously, the repair operator.  FOMO_DEBUG_GENERATOR selects which one to
# test so the difference can be bisected.
policies=(adaptive static_first_cycle)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
policy="${policies[$task]}"
generator="${FOMO_DEBUG_GENERATOR:-strong_augmentation}"
tag="debug_reuse_${policy}_${generator}"

checkpoint="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_0/last.ckpt"
run_root="$BASE_CACHE_DIR/rebuttal_runs/debug_reuse/${policy}_${generator}"
rm -rf "$run_root" "$CHECKPOINT_ROOT_DIR/$tag"
mkdir -p "$run_root"

fomo_wait_for_gpu

python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=false \
    num_runs=1 seed=0 checkpoint="$checkpoint" skip_initial_training=true \
    max_cycles=3 n_epochs_per_cycle=1 max_steps_per_cycle=20 \
    train_batch_size=64 val_batch_size=128 \
    num_ood_samples=20 num_generations_per_ood_sample=2 \
    sample_selection=ood ood_selection_strategy=mode_window \
    selection_reuse_policy="$policy" \
    ood_augmentation=true generation_model="$generator" \
    ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="$tag"
