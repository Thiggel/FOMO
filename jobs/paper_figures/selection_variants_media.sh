#!/bin/sh
#SBATCH --job-name=paper-figures-selection-variants
#SBATCH --output=job_logs/paper_figures/selection_variants_%A_%a.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2

cd $HOME/FOMO

. jobs/environment.sh

mkdir -p job_logs/paper_figures

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant_name=mode_window_q75_diverse
    extra_args="sample_selection=ood ood_selection_strategy=mode_window ood_mode_histogram_quantile_range=[0.75,0.99] ood_mode_candidate_pool_multiplier=4 ood_mode_diversity_sampling=true ood_mode_diversity_normalize_features=true"
    ;;
  1)
    variant_name=top_tail
    extra_args="sample_selection=ood ood_selection_strategy=top"
    ;;
  2)
    variant_name=uniform
    extra_args="sample_selection=uniform"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

additional_path="${BASE_CACHE_DIR}/paper_figures/${variant_name}_seed0"

# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m experiment \
    model=resnet50 \
    ssl=simclr \
    dataset=imagenet100_imbalanced \
    max_cycles=5 \
    n_epochs_per_cycle=100 \
    ood_augmentation=true \
    experiment_name=paper_figures_selection_${variant_name} \
    train_batch_size=512 \
    num_runs=1 \
    seed=0 \
    additional_data_path=${additional_path} \
    enable_media_logging=true \
    save_visualization_data=true \
    log_class_dist=true \
    log_tsne=true \
    log_generated_samples=true \
    save_class_distribution=true \
    logger=true \
    ${extra_args}
