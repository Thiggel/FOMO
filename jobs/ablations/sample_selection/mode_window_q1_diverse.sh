#!/bin/sh
#SBATCH --job-name=ablations-sample-selection-mode-window-q1-diverse
#SBATCH --output=job_logs/ablations/sample_selection/mode_window_q1_diverse_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/ablations/sample_selection

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    sample_selection=ood \
    ood_selection_strategy=mode_window \
    ood_mode_histogram_quantile_range=[0.01,0.99] \
    ood_mode_candidate_pool_multiplier=4 \
    ood_mode_diversity_sampling=true \
    ood_mode_diversity_normalize_features=true \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=ablations_sample-selection_mode-window-q1-diverse \
    num_runs=3 \
    train_batch_size=512
