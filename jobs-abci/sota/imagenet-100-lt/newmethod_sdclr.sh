#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=150:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/imagenet-100-lt

python -m experiment model=resnet50 ssl=sdclr dataset=imagenet100_imbalanced num_cycles=8 total_epochs=800 ood_augmentation=true experiment_name=sota_imagenet-100-lt_newmethod_sdclr train_batch_size=512 use_deepspeed=false log_class_dist=true log_generated_samples=true log_tsne=true num_runs=1 >& job_logs/imagenet-100-lt/newmethod_sdclr.out
