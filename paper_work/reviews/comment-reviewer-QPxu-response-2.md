We thank the reviewer for reading our response and for stating the remaining concerns clearly. We agree that our first response did not explain the principle behind sparse repair formally enough. We address this below and also report the completed full schedule ViT results.

### **[i] Why repairing sparse support can help**

We consider a representation as a collection of locally coherent regions. Let region \(j\) have probability mass \(p_j\), local sample support \(n_j\), and representation error

\[
E_j(n_j)=c_j n_j^{-\alpha}, \qquad \alpha>0.
\]

This is the usual diminishing returns assumption for learning from additional samples. If \(m\) valid samples are added to this region, its contribution to the reduction in error is

\[
\Delta_j(m)=p_jc_j\left[n_j^{-\alpha}-(n_j+m)^{-\alpha}\right].
\]

The marginal value of another sample is therefore

\[
\frac{\partial\Delta_j}{\partial m}
=\alpha p_jc_j(n_j+m)^{-\alpha-1}.
\]

For two otherwise comparable coherent regions, the marginal value is larger in the region with lower local support. Thus, under a finite data acquisition budget, allocating all samples uniformly is inefficient because it spends part of the budget in regions where another sample has little marginal value. This is the principle which motivates using local support in the current representation as the acquisition signal.

This result is deliberately limited. It does not claim a proof of the complete neural SSL training process. It formalizes the condition under which sparse support is the correct place to spend an additional data budget. We will add this derivation and its assumptions to the manuscript.

### **[ii] Why the most extreme tail is excluded**

The argument above assumes that an added sample is a valid sample from the same local region. This need not hold for the most extreme observations. Let \(\rho_j\) denote the probability that the repair operation preserves the region and let \(C_j\) denote the cost of an invalid or semantically drifting repair. The expected utility becomes

\[
U_j(m)=\rho_j\Delta_j(m)-(1-\rho_j)C_j.
\]

Moving from dense to sparse regions increases the possible benefit \(\Delta_j\). In the extreme tail, however, the probability of faithful repair can decrease because this part of the distribution increasingly contains isolated observations, corruptions, or inputs outside reliable support of the generator. If \(\rho_j\) falls sufficiently in this tail, the expected utility reaches its maximum before the most extreme observations. This gives an interior sparse region rather than the farthest OOD samples as the repair target.

The mode window is a nonparametric implementation of this result. Trimming excludes the region in which repairability is least reliable. The mode within the remaining sparse band favors a region with repeated empirical support rather than isolated endpoints. Farthest point sampling then prevents the finite budget from being spent on near duplicates.

The derivation does not claim that the exact 75 to 99 range or \(k=100\) is universally optimal. They are operating parameters used to estimate the sparse but repairable region. Importantly, the result is not sensitive to the precise lower percentile. Over the complete five stage protocol, the linear probe averages are 42.1 for \(q_{1\text{--}99}\), 42.2 for \(q_{50\text{--}99}\), and 42.3 for \(q_{75\text{--}99}\). In contrast, uniform selection obtains 35.5 and direct top tail selection obtains 34.3. The corresponding \(k\) sweep also changes performance only slightly. The same defaults were used without dataset specific tuning for ImageNet 100 LT, CIFAR 10 LT, CIFAR 100 LT, PASS, and DiffusionDB.

We also tested the reported three seed selector results statistically. We used conservative Welch tests based on the reported means and sample standard deviations. Every mode window range remains better than both uniform and top tail selection after Holm correction over the eight selector comparisons, with adjusted \(p\leq .011\). In contrast, none of the three mode window ranges differs detectably from another. This is the statistical result we would expect if excluding the extreme tail is important but the exact lower percentile is not.

We will revise the manuscript to make this distinction explicit. The contribution is the sparse but repairable acquisition principle and not the claim that one numerical percentile interval is optimal for every distribution. We will add the derivation, the complete sensitivity results, and this narrower statement to the manuscript.

### **[iii] MoCo v3 with ViT S under the full schedule**

The reviewer is correct that the one cycle MoCo and DINO diagnostics in our first response were too small to establish a meaningful effect. The full MoCo v3 with ViT S experiment has now completed for three paired runs. Every pair starts from the same source checkpoint and receives the same optimizer updates after branching.

| Result | MoCo v3 | MoCo v3 with BRIDGE | Paired difference | Paired p |
|---|---:|---:|---:|---:|
| Four task average | 24.78 ± 0.77 | **26.16 ± 0.21** | **+1.38 ± 0.89** | .115 |

The paired gains are +0.82, +2.41, and +0.92, so all three completed runs improve under the full schedule. Cars, Aircraft, and ImageNet 100 LT improve in every pair. Flowers is mixed. We report the sample standard deviation throughout and do not describe every target as improved.

We also evaluated a fixed panel containing only original images. Effective rank increases by **21.97 ± 1.58** with paired \(p=.0017\), spectral entropy increases by **0.296 ± 0.019** with paired \(p=.0014\), and 1 NN accuracy increases by **3.80 ± 0.80** with paired \(p=.0146\). Effective rank and spectral entropy remain significant after Holm correction over all eight reported full schedule transfer and geometry endpoints.

With only three paired runs, the aggregate four task transfer difference does not pass a two sided 0.05 test despite being positive in every run. We therefore report the effect size and uncertainty rather than claiming significance which is not supported by the sample size. The completed result nevertheless replaces the very small one cycle diagnostic. Together with the significant representation measurements, it shows that the repeated full schedule has a clear effect with a ViT native SSL learner even though not every downstream target improves.

We will add the full schedule ViT results, individual downstream results, uncertainty, and statistical testing to the manuscript. We will also revise the broader claim. BRIDGE is compatible with different SSL learners, while the size of the gain depends on the objective, architecture, training duration, and downstream target.
