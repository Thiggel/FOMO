#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/selection

python -m experiment model=vit_base ssl=moco dataset=imagenet1k_imbalanced sample_selection=uniform num_cycles=5 total_epochs=100 ood_augmentation=true experiment_name=ablations_selection_moco_uniform train_batch_size=512 use_deepspeed=false >& job_logs/selection/moco_uniform.out
