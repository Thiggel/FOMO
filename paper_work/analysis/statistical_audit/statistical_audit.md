# Statistical audit of rebuttal experiments

The experimental unit is the independent training seed. Tests are two sided paired t tests because the compared runs share seeds and branch checkpoints. The table reports unadjusted p values and Holm adjusted p values within each experimental family. Confidence intervals use the t distribution. With only three seeds the intervals are necessarily wide and the exact Wilcoxon test has very low resolution.

| Family | Comparison | Metric | n | Mean difference | 95% CI | p | Holm p | Cohen dz | Wilcoxon p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | four task average | 3 | 1.383 | [-0.827, 3.592] | 0.1147 | 0.2408 | 1.554 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | Cars accuracy | 3 | 1.542 | [1.238, 1.846] | 0.0021 | 0.0126 | 12.590 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | Aircraft accuracy | 3 | 2.450 | [-0.496, 5.396] | 0.0700 | 0.2408 | 2.066 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | Flowers accuracy | 3 | -0.871 | [-14.095, 12.352] | 0.8034 | 0.8034 | -0.164 | 1.0000 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | ImageNet 100 LT accuracy | 3 | 2.410 | [-0.256, 5.075] | 0.0602 | 0.2408 | 2.245 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | effective rank | 3 | 21.972 | [18.048, 25.896] | 0.0017 | 0.0120 | 13.910 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | spectral entropy | 3 | 0.296 | [0.248, 0.344] | 0.0014 | 0.0113 | 15.358 | 0.2500 |
| full MoCo v3 ViT S | BRIDGE minus MoCo v3 | 1 NN accuracy | 3 | 3.797 | [1.804, 5.789] | 0.0146 | 0.0728 | 4.733 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | four task average | 3 | 2.003 | [-0.300, 4.306] | 0.0646 | 0.3229 | 2.160 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | Cars accuracy | 3 | 0.493 | [-0.586, 1.573] | 0.1882 | 0.3362 | 1.135 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | Aircraft accuracy | 3 | 1.340 | [-0.446, 3.126] | 0.0840 | 0.3362 | 1.864 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | Flowers accuracy | 3 | 4.139 | [-3.870, 12.149] | 0.1562 | 0.3362 | 1.284 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | ImageNet 100 LT accuracy | 3 | 2.039 | [0.636, 3.442] | 0.0246 | 0.1479 | 3.609 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | effective rank | 3 | -6.945 | [-8.251, -5.639] | 0.0019 | 0.0152 | -13.214 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | spectral entropy | 3 | -0.072 | [-0.088, -0.056] | 0.0026 | 0.0182 | -11.294 | 0.2500 |
| full SimCLR ViT S | BRIDGE minus SimCLR ViT S | 1 NN accuracy | 3 | 0.992 | [-0.488, 2.471] | 0.1022 | 0.3362 | 1.665 | 0.2500 |
| full DINO ViT S | BRIDGE minus DINO ViT S | four task average | 2 | -0.784 | [-5.916, 4.349] | 0.3030 | 1.0000 | -1.372 | 0.5000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | Cars accuracy | 2 | -0.267 | [-3.981, 3.446] | 0.5283 | 1.0000 | -0.647 | 1.0000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | Aircraft accuracy | 2 | -0.285 | [-0.857, 0.287] | 0.0997 | 0.6979 | -4.478 | 0.5000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | Flowers accuracy | 2 | -2.614 | [-27.529, 22.300] | 0.4097 | 1.0000 | -0.943 | 0.5000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | ImageNet 100 LT accuracy | 2 | 0.033 | [-1.209, 1.274] | 0.7952 | 1.0000 | 0.236 | 1.0000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | effective rank | 2 | -5.096 | [-46.834, 36.641] | 0.3645 | 1.0000 | -1.097 | 0.5000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | spectral entropy | 2 | -0.076 | [-0.717, 0.564] | 0.3714 | 1.0000 | -1.071 | 0.5000 |
| full DINO ViT S | BRIDGE minus DINO ViT S | 1 NN accuracy | 2 | 1.102 | [0.910, 1.294] | 0.0087 | 0.0697 | 51.640 | 0.5000 |
| short ViT diagnostics | one cycle MoCo BRIDGE minus base | four task average | 3 | 0.530 | [-0.652, 1.711] | 0.1935 | 0.5806 | 1.114 | 0.2500 |
| short ViT diagnostics | three cycle MoCo BRIDGE minus base | four task average | 3 | 0.120 | [-0.405, 0.646] | 0.4279 | 0.8558 | 0.570 | 0.7500 |
| short ViT diagnostics | MAE BRIDGE minus base | four task average | 3 | -0.439 | [-3.906, 3.027] | 0.6402 | 0.8558 | -0.315 | 1.0000 |
| low label fine tuning | BRIDGE minus base at 1 percent labels | ImageNet 100 LT accuracy | 3 | 1.974 | [0.025, 3.922] | 0.0488 | 0.0977 | 2.516 | 0.2500 |
| low label fine tuning | BRIDGE minus base at 10 percent labels | ImageNet 100 LT accuracy | 3 | 3.166 | [-0.952, 7.285] | 0.0805 | 0.0977 | 1.910 | 0.2500 |
| TS factorial | BRIDGE minus base | CIFAR 10 transfer | 3 | 4.893 | [2.218, 7.569] | 0.0158 | 0.0946 | 4.543 | 0.2500 |
| TS factorial | BRIDGE plus TS minus TS | CIFAR 10 transfer | 3 | 9.200 | [3.454, 14.946] | 0.0204 | 0.1021 | 3.977 | 0.2500 |
| TS factorial | BRIDGE plus TS minus BRIDGE | CIFAR 10 transfer | 3 | -0.473 | [-4.797, 3.851] | 0.6840 | 0.6840 | -0.272 | 0.7500 |
| TS factorial | BRIDGE minus base | CIFAR 100 transfer | 3 | 2.520 | [-3.867, 8.907] | 0.2317 | 0.4897 | 0.980 | 0.2500 |
| TS factorial | BRIDGE plus TS minus TS | CIFAR 100 transfer | 3 | 5.770 | [1.995, 9.545] | 0.0223 | 0.1021 | 3.797 | 0.2500 |
| TS factorial | BRIDGE plus TS minus BRIDGE | CIFAR 100 transfer | 3 | -0.920 | [-2.752, 0.912] | 0.1632 | 0.4897 | -1.248 | 0.2500 |
| generator settings | FLUX dev 20 minus 6 steps | four task average | 3 | 0.550 | [-2.188, 3.288] | 0.4787 | 0.4787 | 0.499 | 0.5000 |
| generator settings | FLUX schnell guidance 3 minus guidance 1 | four task average | 3 | 0.764 | [0.344, 1.183] | 0.0159 | 0.0318 | 4.524 | 0.2500 |
| adaptive acquisition | top tail minus adaptive mode window | four task average | 3 | 1.180 | [-0.197, 2.557] | 0.0663 | 0.1327 | 2.128 | 0.2500 |
| adaptive acquisition | dense placebo minus adaptive mode window | four task average | 3 | 0.349 | [-2.517, 3.216] | 0.6522 | 0.6522 | 0.303 | 1.0000 |
| TADA controls | BRIDGE plus TADA minus TADA | four task average | 3 | -0.977 | [-3.819, 1.864] | 0.2770 | 0.2770 | -0.854 | 0.2500 |

## Interpretation rules

A small p value is not evidence that every downstream dataset improves. The primary four task average and each downstream task are reported separately. Holm adjustment is applied only within the named family and is not a substitute for identifying a primary endpoint in advance.

Comparisons absent from this report either have fewer than two paired seeds in the retained result store, have only manuscript level mean and standard deviation summaries, or do not share an auditable protocol. They must not be described as statistically significant without recovering the raw paired seeds.
