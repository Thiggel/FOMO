We thank the reviewer for the detailed review and the extensive list of relevant references. We appreciate that the reviewer finds the downstream performance strong. We agree that the related work section in the submission is too short and that the training schedule and generator controls require clarification.

### **[W1] Novelty and missing citations**

> The method appears close to existing augmentation, dataset selection and industrial data engine pipelines and several important citations are missing.

We agree that oversampling, the kNN distance, SDEdit, FPS and iterative training are not new separately. Our contribution is to use the evolving SSL representation to decide where a fixed generation budget should add support in an unlabeled skewed source.

The closest work differs along the information and feedback axes.

| Family | Acquisition signal | Cyclic | Learner changes next acquisition |
|---|---|---|---|
| DOPING style | Extreme latent rarity | No | No |
| TADA style | Learning difficulty | Usually no | No |
| StableRep and GenRep | Global synthesis | No | No |
| AIDE and data engines | Supervised task failure | Yes | Yes |
| BRIDGE | Label free local support | Yes | Yes |

BRIDGE combines the absence of source labels, local support acquisition, image conditioned addition under a fixed budget and reestimation after each intervention. We will not claim novelty for the components separately.

We will add the reviewer's suggested SMOTE, DOPING, SDEdit, concept cluster pruning, AIDE, Azizi et al., Fan et al., Learning by Noise, GenRep, StableRep, TADA and REPA references. We will also add DiffAug and DA Fusion. This was an omission in the submitted related work.

### **[W2] Comparisons to targeted/generative augmentation**

> The experimental comparison with prior generative augmentation is limited.

Following this comment, we implemented the acquisition rule adaptations inside the same pipeline. This is a diagnostic after one repair rather than a full method ranking. All rows use ImageNet 100 LT, SimCLR with ResNet 50, Cars, Aircraft, Flowers and ImageNet 100 LT transfer, 2,500 SD3 images, 9,700 optimizer updates and three paired seeds.

| Method | Four task avg. |
|---|---|
| Uniform + SD3 | 18.33 ± 0.32 |
| Cluster frequency + SD3 | 18.66 ± 0.10 |
| BRIDGE + SD3 | 18.88 ± 0.58 |
| Extreme score control + SD3 | 19.51 ± 0.22 |
| Label free TADA style control + SD3 | 20.00 ± 0.98 |

These controls show that the acquisition signal matters but do not establish that the mode window is superior after one repair. The submitted repeated setting gives a different ordering. Over five stages the mode window obtains 42.3 ± 0.7 compared with 34.3 ± 1.2 for the top tail and 35.5 ± 1.2 for uniform. We will describe the TADA and extreme score rows as common pipeline adaptations rather than faithful full method reproductions.

### **[W3] Training schedule and computational cost**

> It is unclear whether each cycle is full retraining, and the repeated SSL cost may be large.

BRIDGE does not retrain the encoder from scratch. It warm starts the encoder weights from the previous stage. The generated data are added to the accumulated source and a fresh optimizer schedule is used for each stage. Five training stages contain four interstage repairs. The generation occurs only when another training stage follows so there is no unused final batch.

Because the old real data remain in every stage this is also not the usual sequential task setup where the previous data disappear. Catastrophic forgetting is therefore less direct although synthetic drift remains a real limitation.

The measured costs are shown below.

| Source | SimCLR | BRIDGE | Overhead |
|---|---|---|---|
| ImageNet 100 LT | 16.9 h | 21.8 h | 29% |
| PASS | 10.2 h | 14.9 h | 46% |
| DiffusionDB | 11.6 h | 16.4 h | 41% |

We agree that this is material and should appear in the main discussion. Four repairs add 10,000 images in total which is 100 percent of a 10k source, 10 percent at 100k and 1 percent at one million. The generation cost stays fixed in this regime while the SSL training cost grows with the source. This arithmetic addresses the relative cost but not whether the fixed budget remains effective at a larger scale. Scoring still requires a full source embedding at each repair and larger sources may need a larger repair budget.

The cycle ablation holds the total optimizer update budget and the total number of generated images fixed. It does not compare more data against less data. It compares fewer repairs with more learning between them against more frequent repairs with less learning between them. We will report the optimizer updates, image exposures, restart points and accumulated data size so this matching is explicit.

### **[W4] External generator prior**

> The gain may be explained by distillation from a strong externally trained generator.

We agree with the reviewer that SD3 imports information learned from external data. To separate the direct feature transfer from changing the source support we tested SD3 feature distillation without generated images.

| Method | Four task avg. | 1 NN |
|---|---|---|
| Uniform SD3 | 18.33 ± 0.32 | 28.97 ± 2.11 |
| BRIDGE | 18.88 ± 0.58 | 28.69 ± 1.82 |
| SD3 feature distillation | 18.02 ± 0.47 | 26.27 ± 2.29 |
| Distillation + BRIDGE | 18.87 ± 0.45 | 29.45 ± 2.26 |

The direct feature transfer does not reproduce the BRIDGE result and combining it with BRIDGE gives no further average gain. This distinguishes the feature transfer on existing inputs from changing where the training mass is added. It does not remove the dependence on the semantic and visual priors learned by the external generator and we will state this limitation.

### **[Q1] Choice of k**

> How was k chosen and is it dataset dependent?

We swept `k={10,25,50,100,200}`. The mean 1 NN representation accuracy stays around **28.5 to 28.9** and the differences between the k values are much smaller than the seed variation. The upper cutoffs `{95,97,99,99.5,100}` vary by less than 0.3 points. Thus k=100 is an empirical default for the tested source and not a universal optimum. We will add the complete sweep and discuss a multiscale kNN score as an alternative.

### **[Q2] Empty prompt, guidance, and strength**

> How sensitive is the method to guidance and editing strength, especially with an empty prompt?

| SD3 setting | Values tested | Four task range |
|---|---|---|
| Guidance | 1, 3, 5, 7.5 | 18.42 to 18.88 |
| Strength | .3, .5, .6, .7, .9 | 18.60 to 19.63 |
| Steps | 6, 20 | 18.58 to 19.70 |

Guidance 5 is not uniquely good. We also checked the pipeline directly with one fixed anchor and generation seed at 20 steps and strength .6. Guidance 3, 5 and 7.5 produced byte identical images. Guidance 1 differed only slightly with a pixel MAE of .00162 and a PSNR of 49.67 dB. Thus guidance 5 does not create a distinct stronger conditional signal in our empty prompt setting. The downstream ranges include training variation so we will not call every difference meaningful. We will report the complete curves.

### **[Q3] FLUX comparison**

> Why do the larger FLUX models perform worse than SD3?

The original comparison was not fully matched. With 20 steps FLUX.1-dev improves from **18.23 to 18.78**. FLUX.1-schnell with guidance 3 reaches **19.68** compared with **19.70** for SD3 at 20 steps. Thus the large original gap mostly came from the sampling recipe. We will replace “SD3 is strongest” with the supported conclusion that BRIDGE works with either generator under matched generation quality.

### **[W5] Diffusion ablation and downstream transfer**

> It is unclear whether diffusion is necessary and whether gains extend beyond linear probing/kNN.

The repopulation condition in the submitted paper restores the real images removed when the long tail was constructed. It is an oracle real data experiment rather than duplication. In a thought experiment an ideal procedure could collect new real photographs near the selected concepts. The oracle restoration obtains **18.68 ± 1.74** and targeted SD3 obtains **18.88 ± 0.58**. We do not claim a difference between them. The generated repair approximately matches access to the missing real support in this diagnostic and diffusion is a practical operator when new real samples are unavailable.

Mode window duplication, conventional augmentation and SD3 obtain **18.66, 18.52 and 18.88** in the short matched branch. Ideally several repair operators should work since the selection of the missing support is the central question. SD3 gives additional local variation and a modest extra gain here.

### **Additional experiments on downstream tasks**

With full encoder fine tuning for 100 epochs and three paired seeds BRIDGE improves ImageNet 100 LT with 1 percent of the labels from **14.31 ± 0.33 to 16.29 ± 1.11** and with 10 percent from **21.56 ± 0.75 to 24.72 ± 1.75**. A separate VOC segmentation evaluation improves **10.56 to 12.32 mIoU**. We omit the weaker Cars and Aircraft pilots until the protocol is better tuned. We will add the completed downstream and ViT results to the manuscript.
