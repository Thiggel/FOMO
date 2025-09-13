#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/imagenet-100-lt

python -m experiment model=resnet50 ssl=simclr dataset=imagenet100_imbalanced max_cycles=8 n_epochs_per_cycle=100 ood_augmentation=true experiment_name=sota_imagenet-100-lt_newmethod train_batch_size=512 use_deepspeed=false >& job_logs/imagenet-100-lt/newmethod.out
