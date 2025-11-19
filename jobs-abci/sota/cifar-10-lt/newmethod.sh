#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=150:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/cifar-10-lt

python -m experiment model=resnet50 ssl=simclr dataset=cifar10_imbalanced num_cycles=8 total_epochs=800 ood_augmentation=true experiment_name=sota_cifar-10-lt_newmethod train_batch_size=512 use_deepspeed=false >& job_logs/cifar-10-lt/newmethod.out
