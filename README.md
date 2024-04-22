# FOMO
Self-Supervised Pre-Training on Imbalanced Datasets using OOD Detection and Diffusion-Based Augmentation

# How to work with this repo

## File Structure

├── experiment
│   ├── __main__.py                             # Main file that runs SSL and finetuning
│   ├── dataset                                 # SSL datasets (ImageNet)
│   │   ├── ContrastiveTransformations.py       # Helper for SimCLR transformations
│   │   ├── ImbalancedImageNetDataModule.py     # Imbalanced ImageNet (-100, -1k, ...)
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
│   │   │                                       # (e.g. SimCLR)
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
