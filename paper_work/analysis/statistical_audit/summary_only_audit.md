# Summary only statistical sensitivity analysis

Raw paired seed values were not retained for these comparisons. The tests below therefore use two sided Welch tests reconstructed from the reported means, sample standard deviations, and three runs per condition. This ignores the pairing and should not replace a paired analysis if the original seeds can be recovered. It is included to show what the published summaries alone support.

| Family | Comparison | Metric | Difference | 95% CI | Welch p | Holm p | Cohen d |
|---|---|---|---:|---:|---:|---:|---:|
| main source averages | BRIDGE minus SimCLR on ImageNet 100 LT | seven task linear probe average | 2.700 | [-0.446, 5.846] | 0.0711 | 0.0711 | 2.307 |
| main source averages | BRIDGE minus SimCLR on CIFAR 10 LT | seven task linear probe average | 5.100 | [2.934, 7.266] | 0.0029 | 0.0116 | 5.361 |
| main source averages | BRIDGE minus SimCLR on CIFAR 100 LT | seven task linear probe average | 10.600 | [7.724, 13.476] | 0.0007 | 0.0036 | 8.713 |
| main source averages | BRIDGE minus SimCLR on PASS 10k | seven task linear probe average | 3.100 | [1.008, 5.192] | 0.0152 | 0.0456 | 3.423 |
| main source averages | BRIDGE minus SimCLR on DiffusionDB 10k | seven task linear probe average | 2.400 | [0.528, 4.272] | 0.0242 | 0.0484 | 2.977 |
| selector summary | q1 to 99 minus top tail | seven task linear probe average | 7.800 | [5.343, 10.257] | 0.0017 | 0.0110 | 7.940 |
| selector summary | q1 to 99 minus uniform | seven task linear probe average | 6.600 | [4.143, 9.057] | 0.0029 | 0.0110 | 6.719 |
| selector summary | q50 to 99 minus top tail | seven task linear probe average | 7.900 | [5.419, 10.381] | 0.0011 | 0.0091 | 7.448 |
| selector summary | q50 to 99 minus uniform | seven task linear probe average | 6.700 | [4.219, 9.181] | 0.0020 | 0.0110 | 6.317 |
| selector summary | q75 to 99 minus top tail | seven task linear probe average | 8.000 | [5.543, 10.457] | 0.0016 | 0.0110 | 8.144 |
| selector summary | q75 to 99 minus uniform | seven task linear probe average | 6.800 | [4.343, 9.257] | 0.0026 | 0.0110 | 6.922 |
| selector summary | q1 to 99 minus q75 to 99 | seven task linear probe average | -0.200 | [-1.787, 1.387] | 0.7440 | 1.0000 | -0.286 |
| selector summary | q50 to 99 minus q75 to 99 | seven task linear probe average | -0.100 | [-1.972, 1.772] | 0.8870 | 1.0000 | -0.124 |
| one repair diagnostic summary | mode window SD3 minus uniform SD3 | four task average | 0.550 | [-0.642, 1.742] | 0.2428 | 0.9000 | 1.174 |
| one repair diagnostic summary | mode window SD3 minus oracle real restoration | four task average | 0.200 | [-3.654, 4.054] | 0.8648 | 1.0000 | 0.154 |
| one repair diagnostic summary | extreme score SD3 minus mode window SD3 | four task average | 0.630 | [-0.628, 1.888] | 0.1922 | 0.9000 | 1.436 |
| one repair diagnostic summary | TADA style SD3 minus mode window SD3 | four task average | 1.120 | [-0.885, 3.125] | 0.1800 | 0.9000 | 1.391 |
| one repair diagnostic summary | mode window SD3 minus SD3 feature distillation | four task average | 0.860 | [-0.357, 2.077] | 0.1198 | 0.7186 | 1.629 |
| one repair diagnostic summary | distillation plus BRIDGE minus mode window SD3 | four task average | -0.010 | [-1.216, 1.196] | 0.9824 | 1.0000 | -0.019 |
| generator summary | FLUX dev 20 steps minus FLUX dev 6 steps | four task average | 0.550 | [-2.378, 3.478] | 0.6155 | 1.0000 | 0.449 |
| generator summary | FLUX schnell guidance 3 minus FLUX schnell 20 steps | four task average | 0.830 | [-1.825, 3.485] | 0.4106 | 1.0000 | 0.766 |
| generator summary | SD3 20 steps minus FLUX dev 20 steps | four task average | 0.920 | [-2.388, 4.228] | 0.3818 | 1.0000 | 0.884 |
| generator summary | SD3 20 steps minus FLUX schnell 20 steps | four task average | 0.850 | [-2.121, 3.821] | 0.3758 | 1.0000 | 0.893 |
| generator summary | SD3 20 steps minus FLUX schnell guidance 3 | four task average | 0.020 | [-1.699, 1.739] | 0.9714 | 1.0000 | 0.032 |
| generator summary | SD3 20 steps minus SD3 6 steps | four task average | 1.120 | [-1.437, 3.677] | 0.2255 | 1.0000 | 1.328 |

These p values are sensitivity checks rather than substitutes for the missing paired seed records. In particular, no one repair diagnostic or generator comparison should be called significant from these summaries after familywise correction.
