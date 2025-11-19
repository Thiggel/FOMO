#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/laioncoco-subset

python -m experiment model=resnet50 ssl=simclr dataset=laioncoco_subset num_cycles=1 total_epochs=800 experiment_name=sota_laioncoco-subset_simclr train_batch_size=512 use_deepspeed=false >& job_logs/laioncoco-subset/simclr.out
