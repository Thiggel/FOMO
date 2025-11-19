#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

. jobs-abci/environment.sh

mkdir -p job_logs/diffusiondb-subset

python -m experiment model=resnet50 ssl=simclr dataset=diffusiondb_subset num_cycles=8 total_epochs=800 use_temperature_schedule=true ood_augmentation=true experiment_name=sota_diffusiondb-subset_newmethod_ts train_batch_size=512 use_deepspeed=false >& job_logs/diffusiondb-subset/newmethod_ts.out
