We thank the reviewer for the detailed and constructive feedback. We are encouraged that the reviewer finds the motivation well articulated, the evaluation broad, and the ablations comprehensive. In the following, we address the remaining weaknesses and questions and report the additional experiments.

### **[W1/Q1] Objective and backbone confound**

> The objective ablation fixes the backbone to ResNet 50 although MoCo and DINO are commonly paired with ViTs.

We agree with the reviewer that the ResNet 50 experiment cannot establish that SimCLR is intrinsically better than MoCo or DINO. Following the suggestion, we reran MoCo v3 and DINO with ViT S on the same ImageNet 100 LT source. The table reports a diagnostic average over Cars, Aircraft, Flowers and ImageNet 100 LT. Each paired arm starts from the same source checkpoint and receives the same optimizer updates and data exposures.

| Objective and backbone | Cycles | Base | +BRIDGE | Gain |
|---|---|---|---|---|
| MoCo v3 with ViT S | One cycle | 14.86 | 15.39 | +0.53 |
| MoCo v3 with ViT S | Three cycles | 22.12 | 22.25 | +0.13 |
| DINO with ViT S | One cycle | 9.72 | 9.76 | +0.04 |

These short runs establish that the paired objective and backbone recipes execute under the same budget but the differences are too small to support a claim about the effect size without completed seed variation. We therefore do not treat them as estimates of the final five stage gap. The full paired runs use 100 epochs per stage and common source checkpoints.

We thank the reviewer for pointing out that the original wording was too broad. We will remove “SimCLR is the strongest objective.” The supported conclusion is that BRIDGE is compatible with several SSL learners while the benefit depends on the objective, architecture and training schedule. The SOTA comparison in the submitted paper concerns the completed SimCLR setting and is separate from this diagnostic.

### **[W2/Q2] Additional geometry evidence**

> Please provide per dataset histograms or UMAP and tSNE evidence showing how the sparse region changes.

We agree that Fig. 2 alone is too qualitative. We therefore evaluate a fixed panel. We select the same set of original source images once and encode exactly those images at every cycle checkpoint. Generated images are not placed in this evaluation set. The changes in the measurements therefore reflect changes in the learned representation rather than changes in the evaluated images. The features are L2 normalized. We measure

* median, p90 and p95 local kNN radius
* radius Gini, effective rank and spectral entropy
* 1 NN neighborhood purity.

Using only the original images avoids the trivial effect that density improves merely because generated points were inserted. We also generated the requested per dataset histograms and joint before and after UMAP and tSNE views. These are useful illustrations but the global quantitative metrics are mixed and we do not use the projections as causal evidence. We will include the plots in the revised manuscript together with invariant distance based measurements.

This analysis suggests a wording correction. Our mechanism concerns the local support around the treated regions and does not require every global uniformity statistic to improve monotonically. We will make the change in local support for the selected anchors relative to controls matched on the initial radius our primary analysis and treat the global geometry as secondary.

### **[Q3] FLUX sampling settings**

> Is the FLUX gap partly caused by using fewer denoising steps and lower guidance?

We agree that the original operating points do not isolate the model choice from the sampling budget. We reran FLUX and SD3 with matched settings.

| Generator | Setting | Four task avg. |
|---|---|---|
| FLUX.1-dev | 6 steps | 18.23 ± 0.98 |
| FLUX.1-dev | 20 steps | 18.78 ± 1.43 |
| FLUX.1-schnell | 20 steps | 18.85 ± 1.30 |
| FLUX.1-schnell | guidance 3 | 19.68 ± 0.81 |
| SD3 | 6 steps | 18.58 ± 1.14 |
| SD3 | 20 steps | 19.70 ± 0.35 |

The rows are means and sample standard deviations over three paired seeds on Cars, Aircraft, Flowers and ImageNet 100 LT. The large original FLUX gap mostly disappears. We will therefore not conclude that SD3 is inherently stronger. The supported conclusion is that the downstream utility is sensitive to the sampling recipe and that BRIDGE can use either generator.

### **[Q4] BRIDGE and TS**

> Why does BRIDGE+TS occasionally outperform BRIDGE although TS is weaker by itself?

The Table 3 wins are averages over three seeds and we do not dismiss them as random variation. This is not simply another run of Table 3 on CIFAR 10 LT. We trained Base, TS, BRIDGE and BRIDGE+TS together with the same source and training budget so that the effect of TS and its interaction with BRIDGE can be separated.

| Method | CIFAR 10 transfer | CIFAR 100 transfer |
|---|---|---|
| Base | 59.71 ± 0.99 | 30.19 ± 1.78 |
| TS | 54.93 ± 1.14 | 26.02 ± 0.63 |
| BRIDGE | **64.61 ± 0.26** | **32.71 ± 0.86** |
| BRIDGE+TS | 64.13 ± 1.49 | 31.79 ± 0.90 |

These values are averages over three seeds. BRIDGE+TS remains substantially better than TS but does not improve the overall average over BRIDGE. We therefore do not claim a general synergy. The small wins in Table 3 are target dependent interactions between an optimization change on a fixed distribution and a change to the training support. We will remove any implication that combining the methods is generally preferable.

### **[Q5/W3] Cost and method limitations**

> Please report the practical overhead and discuss limitations of the method itself.

BRIDGE warm starts the encoder weights between cycles and does not retrain the model from scratch. Each stage uses a fresh optimizer schedule. The measured wall clock costs are shown below.

| Source | SimCLR | BRIDGE | Overhead |
|---|---|---|---|
| ImageNet 100 LT | 16.9 h | 21.8 h | 29% |
| PASS | 10.2 h | 14.9 h | 46% |
| DiffusionDB | 11.6 h | 16.4 h | 41% |

We agree that this belongs in the main discussion. A run with five training stages contains four interstage repairs and no unused final generation. Each repair adds 2,500 images so the total is 10,000. This is 100 percent of a 10k source, 10 percent of a 100k source and 1 percent of a one million image source. The generation cost is fixed in this regime while the SSL training cost grows with the source size. This is a cost argument rather than evidence that the fixed budget remains effective at the million image scale. Scoring still requires one embedding pass per repair. We will also discuss the dependence on early encoder quality, the generator bias and coverage, synthetic drift and the possibility that larger sources require a larger repair budget.

### **Additional experiments on downstream tasks**

We additionally tested end to end low label transfer with 100 fine tuning epochs and three paired seeds. With 1 percent of the labels ImageNet 100 LT improves from **14.31 ± 0.33 to 16.29 ± 1.11**. With 10 percent of the labels it improves from **21.56 ± 0.75 to 24.72 ± 1.75**. A separate VOC segmentation evaluation improves from **10.56 to 12.32 mIoU**. We will add these downstream tasks and the completed ViT results to the manuscript.

We will also expand the related work on supervised generative augmentation, synthetic pretraining, targeted diffusion augmentation and data engines. The main distinction is the label free acquisition that is recomputed after the representation changes.
