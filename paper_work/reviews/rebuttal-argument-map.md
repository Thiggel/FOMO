# Rebuttal argument map

This is an internal map, not text to paste directly into OpenReview.

The UMAP/t-SNE files are currently on the author's other device rather than in
this workspace. Recheck the exact qualitative wording against those figures
before posting the responses.

## Reviewer 2x5B

| Review point | Response we make | Evidence used | Boundary of the claim |
|---|---|---|---|
| Weakness / Q1: MoCo and DINO were paired with ResNet-50 | Agree that the original ablation cannot rank objectives intrinsically. Show that BRIDGE also gives a positive paired delta with ViT-native objectives. | Initial MoCo v3/ViT-S: 14.86 -> 15.39. Three-stage MoCo: 22.12 -> 22.25. Initial DINO/ViT-S: 9.72 -> 9.76. | These are compatibility results. Do not claim large or equal gains for every objective. |
| The ViT screens are small | Explain that they contain one repair stage and mainly remove the architecture confound. The paper's cycle ablation shows that the repair effect develops over several cycles. | Existing cycle ablation; full five-stage x 100-epoch MoCo/DINO runs are active. | Say the longer experiment is running. Do not state its outcome in advance. |
| Weakness / Q2: Fig. 2 is insufficient | Define the fixed panel and explain the additional quantitative and visual analysis. | Same original images encoded at each checkpoint; normalized radii, Gini, effective rank, spectral entropy, neighborhood purity; per-dataset histograms and joint UMAP/t-SNE. | Claim local filling of selected sparse neighborhoods. Do not claim every global statistic improves monotonically. |
| Q3: FLUX had an unfairly small sampling budget | Accept the confound and provide matched settings. | FLUX-dev 18.23 at 6 steps, 18.78 at 20; FLUX-schnell guidance 3 reaches 19.68; SD3-20 is 19.70. | Conclude generator choice matters less after matching. Do not retain “SD3 is inherently best.” |
| Q4: Why does BRIDGE+TS sometimes win? | The isolated wins are not stable synergy. TS is optimized for a fixed long tail, while BRIDGE changes the distribution. | CIFAR-10 transfer: Base 59.71, TS 54.93, BRIDGE 64.61, BRIDGE+TS 64.13. CIFAR-100: 30.19, 26.02, 32.71, 31.79. | Say mechanisms can occasionally align, but are not reliably additive. |
| Q5: Runtime | Clarify warm-started encoder training and report actual overhead. | ImageNet-100-LT +29%, PASS +46%, DiffusionDB +41%. | Acknowledge material cost. Do not call it negligible at scale without conditions. |
| Weakness: limitations are too generic | Add method-specific limitations. | Early encoder dependence, generator bias/coverage, recursive synthetic drift, generation cost. | Treat these as actual limitations, not only future work. |

## Reviewer QPxu

| Review point | Response we make | Evidence used | Boundary of the claim |
|---|---|---|---|
| Weakness 1: limited technical novelty / existing components | Do not defend the primitives as new. Define the contribution as an unlabeled cyclic support-allocation problem in which the encoder also determines the next data intervention. | Formal loop `theta_c -> S_c -> D_{c+1}` and closest-prior comparisons. | Avoid “every algorithm combines existing ideas.” It sounds defensive and does not establish our contribution. |
| Weakness 2: no causal analysis | Present common-checkpoint branches matched for updates and added-image count. | No repair 18.86; uniform duplicate 17.24; mode duplicate 18.66; uniform SD3 18.33; mode SD3 18.88; extreme-tail SD3 19.51. | The short experiment supports targeted over uniform allocation. It does not show mode-window universally beats extreme-tail. |
| Q1 / role of diffusion | Separate selection from repair. Diffusion adds local variation, but selection explains much of the gain. | Mode duplicate 18.66 vs mode SD3 18.88; oracle real restoration 18.68. | Reframe diffusion as one repair operator. Do not claim it is always essential. |
| Weakness 3 / Q2: 75–99 is heuristic | Give a support-deficit x repairability objective and an interior-optimum example. Replace histogram binning with a minimum-width score interval. | `a(d) rho(d)`, with `d^alpha exp(-beta d)` maximizing at `alpha/beta`; k/cutoff/metric/alpha sweeps. | This motivates an interior region but does not mathematically derive 75 and 99. Call them robust defaults. |
| Weakness 4: rule-based and not end-to-end | Distinguish adaptive from learned. The nonparametric rule is recomputed from each learned encoder. | Adaptive selection implementation; static/one-shot controls. | Do not claim adaptive is best at every short horizon. Explain why end-to-end selection would need a reward or meta-objective. |
| Weakness 5 / geometry | Use the same original images at every checkpoint and show local neighborhood changes. | Fixed-panel metrics plus existing UMAP/t-SNE showing selected sparse regions filling in. | Keep the claim local if global uniformity metrics are mixed. |
| Weakness 6 / Q4: only linear and kNN | Lead with low-label fine-tuning and segmentation. | ImageNet-100-LT 1%: 14.31 -> 16.29; 10%: 21.56 -> 24.72. Cars 1.83 -> 2.40; Aircraft 3.72 -> 4.43; VOC 10.56 -> 12.32 mIoU. | These answer generalization beyond frozen evaluation. |
| Weakness 7 / Q5: stronger SSL | Add ViT-native MoCo/DINO screens and the DINOv3 teacher control. | MoCo 14.86 -> 15.39; DINO 9.72 -> 9.76; DINOv3 teacher 16.99 -> 18.51; full schedules running. | MAE is mixed and MoCo's longer gain is small. Claim compatibility, not universal improvement. |
| Weakness 8: limited baseline comparison | Compare acquisition and diffusion baselines rather than only generic SSL methods. | TADA-SSL, cluster-frequency, DOPING-style tail, DiffAug, SD3 distillation. | TADA and extreme-tail are strong. Present them honestly. |
| Weakness 9: scalability | Explain the cost decomposition and report ongoing fixed-budget/fixed-fraction scaling. | Runtime measurements and 10k–100k scaling jobs. | Fixed-budget overhead need not scale with N, but embedding and training still do. |

## Reviewer SJHM

| Review point | Response we make | Evidence used | Boundary of the claim |
|---|---|---|---|
| Novelty relative to SMOTE, DOPING, SDEdit, data engines | Concede the primitives and distinguish the problem setting and information flow. BRIDGE is label-free, budgeted, local, and encoder-adaptive. | Closest-work comparison table in the response. | Do not claim the first iterative generative data loop; DiffAug is relevant prior work. |
| Missing citations | Thank the reviewer and list what will be added. | SMOTE, DOPING, SDEdit, cluster pruning, AIDE, Azizi, Fan, Learning by Noise, GenRep, StableRep, TADA, REPA, DiffAug, DA-Fusion. | This was a real omission. Do not argue otherwise. |
| Need direct targeted/generative baselines | Report same-source, same-budget results. | Uniform 18.33; cluster 18.66; BRIDGE 18.88; tail 19.51; TADA 20.00. | Describe BRIDGE as competitive, not dominant or unqualified SOTA. |
| More cycles may only mean more data | Use matched-update continued-training and random-addition controls. | Five stages: BRIDGE 18.91, continued 18.64, random 18.28. At ten stages the advantage saturates. | Supports an intermediate optimum and early stopping, not monotonic benefit. |
| Computational cost / full retraining confusion | State the actual implementation: encoder weights are warm-started, all prior real data remain, optimizer schedule restarts per stage. | Code/protocol audit and measured wall-clock table. | Do not say optimizer state continues. |
| Catastrophic forgetting | Old data are retained, so this is not a standard disjoint sequential-task setup. | Accumulated source at every cycle. | Synthetic drift remains possible and should be acknowledged. |
| External teacher / distillation | Accept that SD3 brings an external prior, then isolate it empirically. | SD3 distillation 18.02/26.27 1-NN; distillation+BRIDGE 18.87/29.45; DINOv3 teacher 16.99 -> 18.51. | Claim complementarity, not absence of an external-data advantage. |
| Q1: choice of k | Show broad stability and stop calling k=100 optimal. | k=10–200 gives about 28.5–28.9 1-NN; cutoff sweep changes <0.3 points. | k remains data-dependent; suggest multi-scale kNN. |
| Q2: empty prompt, guidance, strength | Report the full sensitivity range. | Guidance 18.42–18.88; strength 18.60–19.63; steps 18.58–19.70. | Guidance 5 and strength are defaults, not uniquely optimal. Step count matters more. |
| Q3: FLUX result | Admit that the original settings were mismatched and give the matched result. | FLUX-schnell guidance 3: 19.68; SD3-20: 19.70. | Conclude approximate parity under the best matched settings. |
| Diffusion may not be essential | Show generator-free controls and narrow the claim. | Duplicate 18.66; conventional augmentation 18.52; SD3 18.88. | Diffusion's incremental gain is modest in the short branch. |
| Evaluation beyond frozen transfer | Add low-label fine-tuning and VOC segmentation. | Same downstream numbers as above. | Do not imply broad detection/segmentation coverage from one segmentation task. |
