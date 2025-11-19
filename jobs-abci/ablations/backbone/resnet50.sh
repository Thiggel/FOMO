#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/backbone

python -m experiment model=resnet50 ssl=simclr dataset=imagenet1k_imbalanced max_cycles=5 n_epochs_per_cycle=20 ood_augmentation=true experiment_name=ablations_backbone_resnet50 train_batch_size=512 >& job_logs/backbone/resnet50.out
