### **Additional results with a stronger SSL objective and ViT**

Following the reviewer’s question about stronger SSL objectives, we have now completed three paired runs using MoCo v3 with ViT S under the full schedule. Both arms start from the same source checkpoint and receive the same optimizer updates after branching.

| Dataset | MoCo v3 | MoCo v3 with BRIDGE | Paired gain |
|---|---:|---:|---:|
| Cars | 7.75 ± 0.29 | **9.29 ± 0.32** | **+1.54 ± 0.12** |
| Aircraft | 10.48 ± 0.82 | **12.93 ± 1.78** | **+2.45 ± 1.19** |
| Flowers | **35.73 ± 2.30** | 34.86 ± 3.22 | -0.87 ± 5.32 |
| ImageNet 100 LT | 45.17 ± 0.66 | **47.58 ± 1.73** | **+2.41 ± 1.07** |
| Average over four tasks | 24.78 ± 0.77 | **26.16 ± 0.21** | **+1.38 ± 0.89** |

The average gains in the individual pairs are +0.82, +2.41 and +0.92. BRIDGE improves Cars, Aircraft and ImageNet 100 LT in every pair. Flowers is mixed and its mean difference is small relative to the variation. The improvement is substantially clearer under the repeated full schedule than in our one cycle diagnostic. This supports compatibility with MoCo v3 and ViT S while also showing that the downstream effect can depend on the target.

On a fixed panel containing only original images, effective rank increases from 63.85 ± 2.38 to 85.82 ± 3.10 and spectral entropy increases from 4.16 ± 0.04 to 4.45 ± 0.04. The corresponding 1 NN accuracy improves from 44.83 ± 2.31 to 48.63 ± 2.15. These results provide quantitative evidence for increased representation rank and improved neighborhood quality. We do not interpret the global metrics alone as proof that every sparse region becomes denser.

We hope these additional results address the concerns about stronger SSL learners and representation geometry. We would be grateful if the reviewer could take them into account in the final assessment. If the reviewer feels that an important part remains unresolved, we would appreciate knowing which point and would be happy to clarify it during discussion.
