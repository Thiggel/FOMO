#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=140:00:00
#PBS -P gag51492

cd $HOME/FOMO

mkdir -p job_logs

conda env create -f environment.yml >& job_logs/install_environment.out
