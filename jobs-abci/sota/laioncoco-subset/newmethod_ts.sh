#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/laioncoco-subset

python -m experiment model=resnet50 ssl=simclr dataset=laioncoco_subset max_cycles=8 n_epochs_per_cycle=100 use_temperature_schedule=true ood_augmentation=true experiment_name=sota_laioncoco-subset_newmethod_ts train_batch_size=512 >& job_logs/laioncoco-subset/newmethod_ts.out
