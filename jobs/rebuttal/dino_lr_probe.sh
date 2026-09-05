#!/bin/bash
#SBATCH --job-name=fomo-dinolr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/vol/home-vol2/ml/laitenbf/FOMO_runtime/logs/full_metric_queue/dinolr_%A_%a.out
#
# Is the DINO arm undertrained because of its learning rate?
#
# The paired ViT-S table reports DINO as the one objective BRIDGE does not
# help, but its *baseline* reaches roughly half the transfer accuracy of the
# SimCLR and MoCo v3 baselines under an identical schedule and effective batch
# size.  An arm whose baseline is that far off does not test the question.
#
# conf/ssl/dino.yaml sets lr=1e-3, ten times every other objective in the
# repository (SimCLR, MoCo, SDCLR and supervised all use 1e-4, MAE 1.5e-4).
# The effective batch is not the difference: DINO runs 16 x 8 accumulation and
# MoCo v3 runs 64 x 2, both 128.
#
# Probe the baseline only, one seed, at the inherited 1e-3 against the 1e-4
# the other objectives use.  If 1e-4 lifts the baseline materially it is worth
# rerunning the paired comparison; if it does not, the DINO result stands as
# reported and no further compute is spent on it.

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

lrs=(1e-3 1e-4)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
lr="${lrs[$task]}"
label="$(echo "$lr" | tr -d '-')"
tag="rebuttal_dino_lrprobe_${label}"

checkpoint="$(
  find "$CHECKPOINT_ROOT_DIR" -path \
    "*compat*dino_vits_base*/clane9_imagenet-100/seed_0/last.ckpt" \
    -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
)"
if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
    echo "Missing DINO ViT-S source checkpoint" >&2
    exit 3
fi

result="$CHECKPOINT_ROOT_DIR/$tag/clane9_imagenet-100/seed_0/result.json"
if fomo_has_full_metric_suite "$result"; then
    echo "Complete result already exists at $result; skipping."
    exit 0
fi

run_root="$BASE_CACHE_DIR/rebuttal_runs/dino_lrprobe/${label}"
mkdir -p "$run_root"

# Baseline arm only: no repair, so this measures the encoder the DINO recipe
# produces rather than anything about BRIDGE.
fomo_wait_for_gpu

python -m experiment \
    dataset=imagenet100_imbalanced model=vit_small ssl=dino \
    ssl.lr="$lr" \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed=0 checkpoint="$checkpoint" \
    skip_initial_training=false max_cycles=4 n_epochs_per_cycle=100 \
    train_batch_size=16 grad_acc_steps=8 val_batch_size=256 \
    ood_augmentation=false \
    ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="$tag"
