#!/bin/bash
#SBATCH --job-name=fomo-selgen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#
# Does generative repair beat adding the same volume of data another way?
#
# This is the reviewers' central objection -- that re-population performs close
# to diffusion-based augmentation, so the gains may be nothing but extra data.
# The existing cycle family cannot answer it: its "random_add" arm changes the
# acquisition rule *and* the generator at once (random + conventional
# augmentation) against BRIDGE (mode window + SD3), so the two factors are
# confounded.
#
# Run the full 2x2 instead, holding everything else at the five-cycle default
# with a fixed 9,700 optimizer updates and 2,500 added images:
#
#                     conventional augmentation      Stable Diffusion 3
#   mode window       ood_strongaug                  ood_sd3   (= BRIDGE)
#   random            random_strongaug (= existing)  random_sd3
#
# The ood_sd3 and random_strongaug cells already exist as rebuttal_cycles_c5_*,
# but are rerun here so all four share one launcher and one protocol record.

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

arms=(ood_sd3 random_sd3 ood_strongaug random_strongaug)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed="$((task % 3))"
arm="${arms[$((task / 3))]:?Unknown factorial task $task}"
run_suffix="${FOMO_RUN_SUFFIX:-_selgen_v1}"

# Match the fixed-budget cycle protocol exactly: five repair stages sharing
# 9,700 updates, 100 anchors per stage at 5 variants each for 2,500 images.
repairs=5
stages="$((repairs + 1))"
steps="$((9700 / stages))"
anchors="$((500 / repairs))"

case "$arm" in
    ood_sd3)          selection=ood;    generator=stable_diffusion_3 ;;
    random_sd3)       selection=random; generator=stable_diffusion_3 ;;
    ood_strongaug)    selection=ood;    generator=strong_augmentation ;;
    random_strongaug) selection=random; generator=strong_augmentation ;;
    *) exit 2 ;;
esac

experiment_name="rebuttal_selgen_${arm}${run_suffix}"
result="$CHECKPOINT_ROOT_DIR/$experiment_name/clane9_imagenet-100/seed_${seed}/result.json"
if fomo_has_full_metric_suite "$result"; then
    echo "Complete result already exists at $result; skipping."
    exit 0
fi

run_root="$BASE_CACHE_DIR/rebuttal_runs/selgen/${arm}/seed_${seed}${run_suffix}"
mkdir -p "$run_root"

python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmark_suite=paper_full \
    num_runs=1 seed="$seed" \
    max_cycles="$stages" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
    train_batch_size=128 val_batch_size=256 \
    num_ood_samples="$anchors" num_generations_per_ood_sample=5 \
    sample_selection="$selection" ood_augmentation=true \
    ood_selection_strategy=mode_window \
    generation_model="$generator" ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="$experiment_name"
