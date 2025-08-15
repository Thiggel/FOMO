# FOMO
Self-Supervised Pre-Training on Imbalanced Datasets using OOD Detection and Diffusion-Based Augmentation

# How to work with this repo

## How to run the experiment

```
python experiment [... arguments]
```

*Options*:

```
--root_dir

--model_name {ResNet18,ResNet50,ViTSmall,ViTBase}
--ssl_method {SimCLR,SDCLR}

--lr
--temperature
--use-temperature-schedule
--temperature-min
--temperature-max
--t-max
 
Use `--use-temperature-schedule` to enable a cosine schedule for the temperature. When enabled, adjust it with `--temperature-min`, `--temperature-max`, and `--t-max`.
--weight_decay
--max_epochs

--splits
--batch_size
--early_stopping_patience
--checkpoint
--num_runs
--max_hours_per_run

--logger, --no-logger

--pretrain, --no-pretrain
--finetune, --no-finetune
```

## File Structure

```
├── experiment
│   ├── __main__.py                             # Main file that runs SSL and finetuning
│   ├── dataset                                 # SSL datasets
│   │   ├── ContrastiveTransformations.py       # Helper for SimCLR transformations
│   │   ├── ImbalancedDataModule.py             # Imbalanced datasets (ImageNet, CIFAR, ...)
│   ├── loggers                                 # Tensorboard loggers (e.g. image loggers)
│   ├── models
│   │   ├── FinetuningBenchmarks                # Contains all finetuning benchmarks
│   │   │                                       # (e.g. CIFAR-10 module that loads
│   │   │                                       # dataset and defines the train/val/test loop)
│   │   │
│   │   ├── ModelTypes.py                       # Here, all different models are defined
│   │   │                                       # (e.g. resnet-18, resnet-50, ViT)
│   │   │
│   │   ├── SSLMethods                          # Self-supervised training methods
│   │   │                                       # (e.g. SimCLR, SDCLR)
│   │   │
│   │   ├── SSLTypes.py                         # This file defines all SSL methods
│   │   │                                       # that can be selected in the main script
│   │   │
│   │   ├── backbones                           # All backbones (e.g. ViT)
│   │   │
│   │   ├── losses                              # Define losses (e.g. contrastive) here
│   │   │
│   │   └── metrics                             # Metrics such as OOD-metric
│   │  
│   ├── tests                                   # All tests go here
│   │  
│   └── utils                                   # small utility functions
│                                               # (one function per file)
│
├── job_logs                                    # Write your job scripts so that
│                                               # all logs are saved here
│
├── jobs                                        # All job scripts go here
│   ├── environment.sh                          # Use this script to load env
│                                               # in an interactive session
```

## How do I make changes to this repo?

1. Check out and pull the latest changes from main
2. Create a new branch with a meaningful name (`git checkout -b branchname`). This name should reflect a task in Jira
3. After the feature is complete, create a pull request
4. Check whether all tests pass
5. Wait for (or ask) Filipe to review your code (possible back and forth)
6. Filipe merges your branch into main

## Rules

- Never work directly on main (should not be possible anyways)
- Each piece of functionality should have a test script in `experiment/tests`
    - Refer to the Pytest docs for help
    - Each test-file/directory/function needs to start with 'test_', otherwise Pytest will ignore it
    - One test function per file
    - Multiple tests that belong together should be grouped in a sub-directory
- Always commit small chunks, i.e., only one new function or piece of functionality per commit
- Always do proper work, do not write dirty code because you want to get something done
- Please don't be mad if I ask you to do changes to your pull-request. I need to have an overview of how all features fit together and might therefore ask you to restructure your code according to common design patterns (i.e. one function has only one purpose, don't repeat yourself, etc.)
