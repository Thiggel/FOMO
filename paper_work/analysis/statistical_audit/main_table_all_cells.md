# Summary only statistical sensitivity analysis

Raw paired seed values were not retained for these comparisons. The tests below therefore use two sided Welch tests reconstructed from the reported means, sample standard deviations, and three runs per condition. This ignores the pairing and should not replace a paired analysis if the original seeds can be recovered. It is included to show what the published summaries alone support.

| Family | Comparison | Metric | Difference | 95% CI | Welch p | Holm p | Cohen d |
|---|---|---|---:|---:|---:|---:|---:|
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | CIFAR 10 | 1.900 | [0.825, 2.975] | 0.0128 | 0.0642 | 4.990 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | CIFAR 100 | 2.500 | [1.669, 3.331] | 0.0025 | 0.0178 | 7.906 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | Cars | 2.700 | [-0.069, 5.469] | 0.0526 | 0.1578 | 3.087 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | Aircraft | 2.800 | [1.969, 3.631] | 0.0018 | 0.0147 | 8.854 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | Flowers | 0.900 | [-15.599, 17.399] | 0.8514 | 0.8514 | 0.171 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | Pets | 4.300 | [1.234, 7.366] | 0.0192 | 0.0766 | 3.373 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | ImageNet 100 LT | 3.400 | [2.219, 4.581] | 0.0056 | 0.0335 | 9.430 |
| main table cells ImageNet-100-LT | BRIDGE minus SimCLR on ImageNet-100-LT | seven task average | 2.700 | [-0.446, 5.846] | 0.0711 | 0.1578 | 2.307 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | CIFAR 10 | 10.300 | [9.687, 10.913] | 0.0000 | 0.0000 | 40.400 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | CIFAR 100 | 11.800 | [9.651, 13.949] | 0.0007 | 0.0042 | 15.494 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | Cars | 2.700 | [1.038, 4.362] | 0.0143 | 0.0572 | 4.269 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | Aircraft | 1.700 | [-0.466, 3.866] | 0.0946 | 0.2838 | 1.787 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | Flowers | -1.000 | [-4.276, 2.276] | 0.4288 | 0.4288 | -0.728 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | Pets | 3.400 | [-2.238, 9.038] | 0.1479 | 0.2959 | 1.605 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | ImageNet 100 LT | 7.000 | [5.508, 8.492] | 0.0002 | 0.0016 | 10.738 |
| main table cells CIFAR-10-LT | BRIDGE minus SimCLR on CIFAR-10-LT | seven task average | 5.100 | [2.934, 7.266] | 0.0029 | 0.0145 | 5.361 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | CIFAR 10 | 15.200 | [11.878, 18.522] | 0.0014 | 0.0057 | 13.595 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | CIFAR 100 | 17.000 | [13.308, 20.692] | 0.0018 | 0.0057 | 14.577 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | Cars | 4.200 | [1.949, 6.451] | 0.0127 | 0.0212 | 5.689 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | Aircraft | 7.000 | [4.950, 9.050] | 0.0010 | 0.0051 | 8.110 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | Flowers | 11.800 | [4.845, 18.755] | 0.0106 | 0.0212 | 4.065 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | Pets | 7.500 | [5.450, 9.550] | 0.0008 | 0.0051 | 8.689 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | ImageNet 100 LT | 11.100 | [8.340, 13.860] | 0.0004 | 0.0032 | 9.218 |
| main table cells CIFAR-100-LT | BRIDGE minus SimCLR on CIFAR-100-LT | seven task average | 10.600 | [7.724, 13.476] | 0.0007 | 0.0051 | 8.713 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | CIFAR 10 | 1.600 | [0.987, 2.213] | 0.0026 | 0.0212 | 6.276 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | CIFAR 100 | 1.600 | [0.554, 2.646] | 0.0137 | 0.0685 | 3.534 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | Cars | 2.600 | [1.013, 4.187] | 0.0172 | 0.0685 | 5.051 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | Aircraft | 3.000 | [0.091, 5.909] | 0.0464 | 0.0928 | 2.711 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | Flowers | 4.400 | [-4.495, 13.295] | 0.1894 | 0.1894 | 1.479 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | Pets | 3.800 | [1.928, 5.672] | 0.0053 | 0.0321 | 4.713 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | ImageNet 100 LT | 4.400 | [2.551, 6.249] | 0.0044 | 0.0309 | 6.044 |
| main table cells PASS-10k | BRIDGE minus SimCLR on PASS-10k | seven task average | 3.100 | [1.008, 5.192] | 0.0152 | 0.0685 | 3.423 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | CIFAR 10 | 1.500 | [0.454, 2.546] | 0.0169 | 0.1183 | 3.313 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | CIFAR 100 | 1.300 | [0.225, 2.375] | 0.0323 | 0.1290 | 3.414 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | Cars | 1.900 | [-0.150, 3.950] | 0.0612 | 0.1835 | 2.201 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | Aircraft | 2.400 | [0.738, 4.062] | 0.0196 | 0.1183 | 3.795 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | Flowers | 4.500 | [-0.548, 9.548] | 0.0651 | 0.1835 | 2.411 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | Pets | 1.600 | [-1.056, 4.256] | 0.1580 | 0.1835 | 1.482 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | ImageNet 100 LT | 3.800 | [2.667, 4.933] | 0.0007 | 0.0059 | 7.600 |
| main table cells DiffusionDB-10k | BRIDGE minus SimCLR on DiffusionDB-10k | seven task average | 2.400 | [0.528, 4.272] | 0.0242 | 0.1209 | 2.977 |

These p values are sensitivity checks rather than substitutes for the missing paired seed records. In particular, no one repair diagnostic or generator comparison should be called significant from these summaries after familywise correction.
