# Closest-prior diffusion augmentation comparison

All accuracy entries are mean ± sample standard deviation over paired seeds. The transfer average uses Cars, Aircraft, Flowers, and ImageNet-100-LT, which are available for every method.

| Method | Seeds | Four-task transfer (%) | Representation 1-NN (%) | GPU-hours | Scope |
|---|---:|---:|---:|---:|---|
| Uniform + SD3 | 3 | 18.33 $\pm$ 0.32 | 28.97 $\pm$ 2.11 | 4.35 ± 0.15 | Generic diffusion augmentation; matched SD3 volume |
| Top-tail + SD3 | 3 | 19.51 $\pm$ 0.22 | 28.63 $\pm$ 1.48 | 4.44 ± 0.12 | DOPING-inspired rare/extreme-region acquisition |
| Cluster-frequency + SD3 | 3 | 18.66 $\pm$ 0.10 | 28.66 $\pm$ 2.42 | 3.53 ± 0.49 | Inverse cluster-occupancy acquisition |
| TADA-SSL + SD3 | 3 | 20.00 $\pm$ 0.98 | 28.78 $\pm$ 2.22 | 2.46 ± 0.11 | Label-free SSL learning-difficulty adaptation of TADA |
| BRIDGE + SD3 | 3 | 18.88 $\pm$ 0.58 | 28.69 $\pm$ 1.82 | 4.42 ± 0.12 | Mode-window sparse-support acquisition |
| Mode-window + conventional augmentation | 3 | 18.52 $\pm$ 0.65 | 28.91 $\pm$ 1.96 | 4.15 ± 0.11 | Generator-free repair control |
| DiffAug adaptation | 3 | 11.50 $\pm$ 1.87 | 19.39 $\pm$ 2.78 | 2.18 ± 0.11 | Image-SSL adaptation; not an official visual recipe |
| DiffAug adaptation + BRIDGE | 3 | 10.42 $\pm$ 2.17 | 20.28 $\pm$ 2.64 | 2.05 ± 0.23 | Image-SSL adaptation with targeted SD3 repair |
| SD3 feature distillation | 3 | 18.02 $\pm$ 0.47 | 26.27 $\pm$ 2.29 | 3.62 ± 0.09 | External diffusion-prior control without generated images |
| SD3 feature distillation + BRIDGE | 3 | 18.87 $\pm$ 0.45 | 29.45 $\pm$ 2.26 | 3.75 ± 0.11 | External diffusion prior plus targeted source repair |
