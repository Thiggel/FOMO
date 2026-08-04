#!/bin/bash
#SBATCH --job-name=fomo-factorial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-23

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

conditions=(
    no_repair
    uniform_duplicate
    mode_duplicate
    uniform_strongaug
    mode_strongaug
    uniform_sd3
    mode_sd3
    top_sd3
)
task="${SLURM_ARRAY_TASK_ID}"
seed="$((task % 3))"
condition="${conditions[$((task / 3))]}"
checkpoint_file="$CHECKPOINT_ROOT_DIR/rebuttal_branch_source/clane9_imagenet-100/seed_$seed/branch_checkpoint.txt"
checkpoint="$(cat "$checkpoint_file")"

selection=ood
strategy=mode_window
augment=true
remove_diffusion=false
generator=stable_diffusion_3
case "$condition" in
    no_repair) augment=false ;;
    uniform_duplicate) selection=random; remove_diffusion=true ;;
    mode_duplicate) remove_diffusion=true ;;
    uniform_strongaug) selection=random; generator=strong_augmentation ;;
    mode_strongaug) generator=strong_augmentation ;;
    uniform_sd3) selection=random ;;
    mode_sd3) ;;
    top_sd3) strategy=top ;;
esac

run_root="$BASE_CACHE_DIR/rebuttal_runs/causal_factorial/$condition/seed_$seed"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmarks='[CarsFineTune,AircraftFineTune,FlowersFineTune,PetsFineTune,CIFAR10FineTuner,CIFAR100FineTuner,ImageNet100LTFineTune]' \
    num_runs=1 seed="$seed" checkpoint="$checkpoint" \
    skip_initial_training=true max_cycles=2 n_epochs_per_cycle=100 \
    max_steps_per_cycle=4850 \
    train_batch_size=128 val_batch_size=256 \
    num_ood_samples=500 num_generations_per_ood_sample=5 \
    sample_selection="$selection" ood_selection_strategy="$strategy" \
    ood_distance_metric=normalized_l2 \
    ood_augmentation="$augment" remove_diffusion="$remove_diffusion" \
    generation_model="$generator" \
    additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_factorial_${condition}"

