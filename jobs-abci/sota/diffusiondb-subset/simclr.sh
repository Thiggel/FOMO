#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/diffusiondb-subset

python -m experiment model=resnet50 ssl=simclr dataset=diffusiondb_subset max_cycles=1 n_epochs_per_cycle=800 experiment_name=sota_diffusiondb-subset_simclr train_batch_size=512 >& job_logs/diffusiondb-subset/simclr.out
