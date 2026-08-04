#!/bin/bash
#SBATCH --job-name=fomo-cycles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-44

set -euo pipefail
cd "${FOMO_REPO_DIR:?FOMO_REPO_DIR must point at the staged repository}"
. jobs/rebuttal/load_cluster_environment.sh

# Five repair-round counts x three controls x three paired seeds.  There is one
# initial training stage plus C repair stages; the total update budget (9700)
# and total added-image budget (2500) are held fixed.
cycle_values=(1 2 3 5 10)
controls=(no_repair random_add bridge)
task="${SLURM_ARRAY_TASK_ID}"
run_suffix="${FOMO_RUN_SUFFIX:-}"
seed="$((task % 3))"
cell="$((task / 3))"
repairs="${cycle_values[$((cell / 3))]}"
control="${controls[$((cell % 3))]}"
stages="$((repairs + 1))"
steps="$((9700 / stages))"
anchors="$((500 / repairs))"
if [ "$anchors" -lt 1 ]; then anchors=1; fi

augment=true
selection=ood
generator=strong_augmentation
case "$control" in
    no_repair) augment=false ;;
    random_add) selection=random ;;
    bridge) generator=stable_diffusion_3 ;;
esac

run_root="$BASE_CACHE_DIR/rebuttal_runs/fixed_cycles/c${repairs}_${control}/seed_${seed}${run_suffix}"
mkdir -p "$run_root"
python -m experiment \
    dataset=imagenet100_imbalanced model=resnet50 ssl=simclr \
    logger=false pretrain=true finetune=true \
    finetune_benchmarks='[CarsFineTune,AircraftFineTune,ImageNet100LTFineTune]' \
    num_runs=1 seed="$seed" \
    max_cycles="$stages" n_epochs_per_cycle=100 max_steps_per_cycle="$steps" \
    train_batch_size=128 val_batch_size=256 \
    num_ood_samples="$anchors" num_generations_per_ood_sample=5 \
    sample_selection="$selection" ood_augmentation="$augment" \
    generation_model="$generator" ood_distance_metric=normalized_l2 \
    additional_data_path="$run_root/generated" \
    experiment_name="rebuttal_cycles_c${repairs}_${control}${run_suffix}"
