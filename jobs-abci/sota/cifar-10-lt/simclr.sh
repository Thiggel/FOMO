#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=2-0
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

python -m experiment model=resnet50 ssl=simclr dataset=cifar10_imbalanced max_cycles=1 n_epochs_per_cycle=800 experiment_name=sota_cifar-10-lt_simclr train_batch_size=512 use_deepspeed=false
