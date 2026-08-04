### **Additional full schedule ViT results**

We wanted to report an update on the objective and backbone experiment suggested by the reviewer. We have now completed three paired runs using MoCo v3 with ViT S under the full schedule. Each pair starts from the same source checkpoint and receives the same optimizer updates after branching.

| Dataset | MoCo v3 | MoCo v3 with BRIDGE | Paired gain |
|---|---:|---:|---:|
| Cars | 7.75 ± 0.29 | **9.29 ± 0.32** | **+1.54 ± 0.12** |
| Aircraft | 10.48 ± 0.82 | **12.93 ± 1.78** | **+2.45 ± 1.19** |
| Flowers | **35.73 ± 2.30** | 34.86 ± 3.22 | -0.87 ± 5.32 |
| ImageNet 100 LT | 45.17 ± 0.66 | **47.58 ± 1.73** | **+2.41 ± 1.07** |
| Average over four tasks | 24.78 ± 0.77 | **26.16 ± 0.21** | **+1.38 ± 0.89** |

The values are means and sample standard deviations over three paired runs. The average gains in the individual pairs are +0.82, +2.41 and +0.92. BRIDGE improves Cars, Aircraft and ImageNet 100 LT in every pair. Flowers is mixed and its mean difference is small relative to the variation, so we do not claim that every downstream target improves. The full schedule result is considerably clearer than the earlier one cycle diagnostic and provides evidence that the benefit is not limited to the SimCLR and ResNet 50 pairing.

We also evaluated the fixed panel of original images requested by the reviewer. Generated images are excluded from this panel.

| Metric | MoCo v3 | MoCo v3 with BRIDGE | Paired change |
|---|---:|---:|---:|
| Effective rank | 63.85 ± 2.38 | **85.82 ± 3.10** | **+21.97 ± 1.58** |
| Spectral entropy | 4.16 ± 0.04 | **4.45 ± 0.04** | **+0.30 ± 0.02** |
| 1 NN accuracy | 44.83 ± 2.31 | **48.63 ± 2.15** | **+3.80 ± 0.80** |

BRIDGE produces a representation of consistently higher rank with better neighborhood accuracy across all three runs. We treat this as quantitative evidence about representation diversity and neighborhood quality. We do not use these global measurements alone as proof that every sparse region becomes denser. The local selected anchor comparison and the UMAP and tSNE analysis address that more specific question.

We hope these results directly address the objective and backbone concern and would be grateful if the reviewer could take them into account in the final assessment. If anything remains unclear, we would be happy to clarify it during discussion.
