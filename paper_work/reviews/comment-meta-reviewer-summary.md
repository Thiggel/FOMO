We thank the Area Chair and reviewers for the detailed feedback. We concentrated the additional work on the two revisions in the meta review.

### **1. Contribution relative to prior work**

We agree that kNN distance, image editing and farthest point sampling are established components. Our contribution is not one of these components in isolation. BRIDGE formulates label free data repair as a feedback problem. The current SSL representation allocates a fixed data acquisition budget, the resulting intervention changes the representation, and the acquisition signal is then recomputed. This distinguishes BRIDGE from fixed data SSL objectives, global synthetic pretraining, supervised targeted augmentation and one shot data selection.

We have also formalized the principle behind the acquisition rule. Under diminishing returns from local sample support, adding another valid sample has greater marginal value in a region with fewer samples. Directly selecting the sparsest observations is nevertheless undesirable when repair fidelity decreases in the extreme tail. Expected repair utility is then the support gain weighted by the probability of a faithful repair, minus the cost of an invalid repair. If fidelity falls sufficiently in the extreme tail, this utility is maximized in an interior sparse region. BRIDGE uses the mode window as a nonparametric approximation to this sparse but repairable region and uses FPS to avoid spending the budget on near duplicates.

This argument does not derive 75 to 99 or $k=100$ as universal optima. They are operating parameters rather than the contribution. The empirical result matches this distinction. The three tested mode windows obtain 42.1 ± 0.7, 42.2 ± 0.9 and 42.3 ± 0.7, while uniform obtains 35.5 ± 1.2 and direct top tail selection obtains 34.3 ± 1.2. Every mode window remains better than uniform and top tail after Holm correction with adjusted $p\leq .011$, while the three mode windows do not differ detectably from one another.

We will revise the novelty discussion around this feedback formulation and sparse but repairable acquisition principle. We will also compare the closest work by supervision, acquisition signal and whether the learner changes the next acquisition.

### **2. New backbones, robustness and representation analysis**

We completed the requested full schedule ViT experiments. All arms start from paired source checkpoints and receive the same post branch optimizer updates.

| Objective and backbone | Base | With BRIDGE | Paired gain |
|---|---:|---:|---:|
| SimCLR with ViT S | 22.57 ± 0.94 | **24.58 ± 0.85** | **+2.00 ± 0.93** |
| MoCo v3 with ViT S | 24.78 ± 0.77 | **26.16 ± 0.21** | **+1.38 ± 0.89** |

The values average four downstream tasks over three paired runs. All six paired averages improve. SimCLR isolates the backbone change, while MoCo v3 also changes the SSL objective. DINO is mixed, so we will state that the effect generalizes to the completed SimCLR and MoCo v3 settings but depends on the learner and schedule.

We also completed the fixed panel representation analysis using only original images. For MoCo v3 with ViT S, effective rank increases by 21.97 ± 1.58 with paired $p=.0017$, spectral entropy increases by 0.296 ± 0.019 with paired $p=.0014$, and 1 NN accuracy increases by 3.80 ± 0.80 with paired $p=.0146$. Effective rank and spectral entropy remain significant after Holm correction. SimCLR with ViT S improves transfer while its global rank decreases. We will therefore treat rank as an objective dependent diagnostic rather than claim that increasing global rank is the universal mechanism. The main geometric claim will concern local support and neighborhood quality around the selected regions.

The robustness study now covers $k$, percentile cutoffs, distance metrics, candidate pool size, FPS compared with random selection, generator strength, guidance and denoising steps. The exact lower percentile and $k$ have small effects compared with the gap to uniform and extreme tail selection. These results support a robust sparse band rather than one narrowly tuned numerical setting.

### **Additional experiments and clarifications**

Matched FLUX and SD3 experiments show that the original generator ranking was confounded by sampling settings, so we will remove the claim that SD3 is intrinsically stronger. Full encoder fine tuning improves from 14.31 ± 0.33 to 16.29 ± 1.11 with 1 percent of the ImageNet 100 LT labels and from 21.56 ± 0.75 to 24.72 ± 1.75 with 10 percent. VOC segmentation improves from 10.56 to 12.32 mIoU. We also clarified that BRIDGE uses continued pretraining and reported its measured overhead.

Several full schedule experiments completed late in the discussion period, so we summarize them here for consideration in the final assessment. We hope these results and the corresponding narrowing of our claims address the two central points in the meta review.
