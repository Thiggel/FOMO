#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/cifar-10-lt

python -m experiment model=resnet50 ssl=sdclr dataset=cifar10_imbalanced max_cycles=8 n_epochs_per_cycle=100 ood_augmentation=true experiment_name=sota_cifar-10-lt_newmethod_sdclr train_batch_size=512 use_deepspeed=false >& job_logs/cifar-10-lt/newmethod_sdclr.out
