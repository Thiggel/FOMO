We thank the reviewer for the detailed feedback. We are encouraged that the reviewer finds the closed loop idea meaningful, the mode window mechanism interesting and the experiments comprehensive. In the following we address the concerns regarding novelty, the causal mechanism, the selector and the downstream evaluation.

### **[W1] Technical contribution**

> The method combines existing components and has limited algorithmic novelty.

We agree that the kNN distance, diffusion editing and farthest point sampling are established components and we do not claim novelty for them individually. BRIDGE concerns label free acquisition in a feedback loop. The current SSL representation decides where a fixed data budget is spent and this signal is recomputed after the intervention changes the representation.

| Family | Labels or task loss | Adds data | Local acquisition | Recomputed |
|---|---|---|---|---|
| Long tailed SSL objectives | No | No | No | No |
| Global synthetic SSL | No | Yes | No | No |
| Targeted supervised augmentation | Usually | Yes | Sometimes | Usually no |
| BRIDGE | No | Yes | Yes | Yes |

BRIDGE asks where the limited repair budget should go when the labels and minority classes are unknown. The novelty claim concerns the four properties together rather than any component or exact percentile.

### **[W2/Q1] Causal controls and the role of diffusion**

> There is no causal analysis and diffusion only slightly improves over repopulation.

Following the reviewer's suggestion, all branches start from common paired source checkpoints and use the same number of added images and optimizer updates. All branches use continued pretraining. The encoder weights are carried from one cycle into the next cycle and no branch retrains the model from scratch.

The term repopulation in the submitted paper did not mean exact duplication. It restores real images that were removed when the long tail was constructed and is therefore an oracle real data experiment. An ideal repair procedure could send someone to collect more real photographs near the selected concepts. The oracle restoration asks what would happen in that thought experiment. SD3 reaches 18.88 ± 0.58 compared with 18.68 ± 1.74 for the oracle real restoration. We do not claim a difference between them. The result shows that the generated repair can approximately match access to the missing real support in this diagnostic.

The one cycle diagnostic should not be read as proof of the full cyclic method. Mode window SD3 is 18.88 ± 0.58 and uniform SD3 is 18.33 ± 0.32 while the extreme score control is 19.51 ± 0.22. In the submitted full five stage experiment the ordering changes substantially. The mode window reaches 42.3 ± 0.7 over seven tasks compared with 35.5 ± 1.2 for uniform and 34.3 ± 1.2 for the top tail. This suggests that avoiding the extreme tail matters over repeated repair rather than necessarily at the first repair.

We will describe diffusion as a practical way to obtain local variations when the missing real images are unavailable rather than as the sole source of the benefit. We will also distinguish exact duplication, conventional augmentation and oracle real restoration.

### **[W3/Q2] Mode window selection**

> The 75th to 99th percentile mode window is heuristic and lacks justification.

We agree that the paper should not describe 75 to 99 as a theoretically optimal range. The submitted three seed results already show that the lower cutoff is not important. The ranges 1 to 99, 50 to 99 and 75 to 99 obtain 42.1 ± 0.7, 42.2 ± 0.9 and 42.3 ± 0.7. The robust design choice is diversity aware selection inside a non extreme sparse band rather than the exact lower percentile.

We swept `k={10,25,50,100,200}`, the upper cutoffs `{95,97,99,99.5,100}`, the distance metrics, the candidate multipliers `{1,2,4,8}` and FPS compared with random selection. The 1 NN representation accuracy varies by about 0.4 points across k and the cutoff sweep varies by less than 0.3 points. Therefore k=100 and 75 to 99 are empirical defaults rather than claimed optima. We keep the submitted histogram method unchanged and treat a bin free alternative only as a future simplification.

### **[W4] Nonparametric rather than jointly learned selection**

> The method is rule based and scoring, selection and generation steps are not learned jointly.

BRIDGE is adaptive but nonparametric. Those are not contradictory. The acquisition rule is recomputed from the current learned representation every cycle. We chose a modular nonparametric rule to preserve the label free operation and make the repair operator interchangeable. Learning the acquisition policy from a self supervised reward is a promising extension and may improve the current rule.

### **[W5/Q3] Representation geometry**

> The geometric interpretation is not supported by direct measurements.

We use a fixed panel where the same original source images are encoded at every checkpoint without adding the generated images to the evaluation set. The metrics include the normalized local radius, radius Gini, effective rank, spectral entropy and neighborhood purity. The visualizations are illustrative and the global metrics are mixed. We therefore narrow the claim to the local support near the selected regions and will not infer global geometric balancing. The decisive comparison is the change for the selected anchors relative to untreated controls matched on the initial radius.

### **[W6] Scalability**

The nontraining work consists of one full source embedding pass and a fixed generation budget at each repair. In the setting with five training stages, four repairs add 10,000 images. This is 100 percent of a 10k source, 10 percent of a 100k source and 1 percent of a one million image source. This only explains the relative cost under a fixed budget. It does not establish that the same budget remains effective at the million image scale and larger sources may require the budget to grow.

### **[W7/Q4] Fine-tuning and other downstream tasks**

> Evaluation is limited to linear probing and kNN.

| Evaluation | Base | BRIDGE |
|---|---|---|
| ImageNet 100 LT, 1% labels | 14.31 ± 0.33 | **16.29 ± 1.11** |
| ImageNet 100 LT, 10% labels | 21.56 ± 0.75 | **24.72 ± 1.75** |
| VOC segmentation mIoU | 10.56 | **12.32** |

The ImageNet values use full encoder fine tuning for 100 epochs and three paired seeds. These results show that the gain is not limited to linear probing or kNN. We omit the weaker Cars and Aircraft pilots until their protocol is better tuned.

### **[W8/Q5] Stronger SSL objectives**

> Stronger SSL methods such as DINO/MAE and their usual backbones are missing.

The initial paired ViT diagnostics are shown below.

| Objective/backbone | Base | +BRIDGE |
|---|---|---|
| MoCo v3 with ViT S | 14.86 | **15.39** |
| DINO with ViT S | 9.72 | **9.76** |

These are small one cycle runs and the differences are too small to interpret without full seed uncertainty. The completed three cycle MoCo values of **22.12 and 22.25** are likewise not evidence of a detectable improvement. We therefore narrow the claim from objective agnostic improvement to compatibility whose benefit depends on the learner and schedule.

We also added label free TADA, cluster frequency, DOPING style extreme tail, DiffAug and diffusion feature distillation controls. These are the most relevant comparisons because they test the acquisition signal and the external generator explanation directly.

We will add the completed ViT experiments and the new downstream tasks to the manuscript.
