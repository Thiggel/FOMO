# FOMO

Anonymous code release for the paper:

`BRIDGE: Balancing Representations by Identifying and Generating Underrepresented Data`

This repository contains the training code, experiment configurations, job scripts, and paper assets used for the BRIDGE experiments.

## Overview

BRIDGE is a self-supervised learning pipeline that alternates between:

1. SSL pretraining on the current source dataset
2. sparsity scoring in representation space with mean $k$NN distance
3. targeted image-to-image augmentation of underrepresented regions

The repository supports:

- baseline runs on balanced and imbalanced sources
- BRIDGE runs with SimCLR, TS, and SDCLR
- ablations over SSL objective, generation method, selection rule, cycle count, and architecture
- source-regime sweeps on ImageNet-100-LT, CIFAR-10-LT, CIFAR-100-LT, PASS, and DiffusionDB

## Repository Structure

```text
experiment/                  Core training, datasets, models, and evaluation code
experiment/conf/             Hydra configuration files
jobs/                        SLURM job scripts for baselines, ablations, and source-regime sweeps
scripts/                     Utility scripts for log checking, result export, and paper table generation
paper_work/                  Manuscript sources, generated tables, and figures
job_logs/                    Collected SLURM logs from completed runs
outputs/                     Experiment outputs and checkpoints
visualizations/              Auxiliary visualizations
```

## Environment

The project expects a Python environment with the packages in:

- `requirements.txt`
- `environment.yml`

The codebase uses Hydra for configuration management and is designed to run both locally and through SLURM job arrays.

## Running Experiments

The main entry point is:

```bash
python -m experiment
```

Typical arguments include:

```text
model_name={ResNet18,ResNet50,ViTSmall,ViTBase}
ssl_method={SimCLR,SDCLR,MoCo,DINO}
dataset=...
pretrain=true
finetune=true
logger=true
num_runs=3
seed=...
```

For SLURM arrays, each task runs one seed through `SLURM_ARRAY_TASK_ID`. After all seed jobs finish, aggregate with:

```bash
python -m experiment aggregate_only=true ...
```

using the same experiment name and configuration.

## Reproducing Paper Results

The repository includes:

- experiment configs under `experiment/conf/`
- job scripts under `jobs/`
- log summaries under `job_logs/`
- result-export and paper-table utilities under `scripts/`

The manuscript in `paper_work/neurips_bridge_paper/` is generated from the completed three-seed runs reported in the paper.

## Notes

- PASS and DiffusionDB source-regime comparisons use aligned 10k source subsets.
- The main reported BRIDGE configuration uses ResNet-50, 5 cycles, 500 selected samples per cycle, and 5 generated images per selected sample.
- The DINO ablation uses four GPUs and two local crops.
