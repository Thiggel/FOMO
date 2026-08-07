# Reviewer risk audit for the ICLR revision

## Main table structure

1. Tables 1 and 2 contain the five source-regime comparisons and report every downstream dataset directly.
2. Table 3 tests generalization beyond the default protocol. It contains paired full-schedule ViT-S results with SimCLR and MoCo v3, low-label full fine-tuning, and PASCAL VOC segmentation.
3. Table 4 contains the core selector and cycle evidence. It shows that several mode-window ranges behave similarly, while uniform and top-tail acquisition are substantially weaker over five cycles.
4. The fixed-space anchor-versus-control table is an appendix diagnostic. It verifies where generated support is inserted but is not presented as proof of the learned representation mechanism.

The unresolved balanced-versus-imbalanced sanity table was removed. Its SimCLR result conflicted with the source-regime result even though the launch scripts appeared equivalent.

## Length of related work

The main related-work section is intentionally compact and organized into four families. The detailed method-by-method comparison is in the appendix. This is necessary because the earlier paper omitted several close areas named by Reviewer SJHM. Returning to the earlier short citation list would repeat that weakness. Keeping the taxonomy table in the appendix prevents the literature audit from overwhelming the main argument.

## Theory and evidence

| Statement | Status | Required evidence |
|---|---|---|
| Adding valid points cannot increase the local kNN radius | Proven without a statistical assumption | Fixed-space anchor-versus-control analysis checks that generated samples enter treated neighborhoods in practice |
| Under the regional learning curve in Eq. 2, lower-support regions have greater marginal value | Proven conditional on the stated learning-curve model | Mode-window versus uniform acquisition tests the model's qualitative prediction |
| Recoverability-weighted utility has a unique interior optimum under the derivative conditions in Proposition 3 | Proven conditional on increasing support value, decreasing repairability, and the single-crossing condition | Mode-window versus top-tail tests the interior-versus-extreme prediction; percentile sweeps show that the exact lower cutoff is not special |
| Greedy adaptive allocation is optimal for separable discretely concave regional utilities | Proven conditional on separability and diminishing returns | Adaptive-versus-frozen and one-shot experiments are needed to establish that this model describes the BRIDGE setting |
| Farthest-first gives a factor-two approximation for metric k-center | Established result from Gonzalez (1985) | FPS-versus-random selection tests whether the coverage argument matters empirically |

The paper must never say that the proofs establish an unconditional improvement for a deep SSL model. They justify a policy class under explicit assumptions. The experiments test whether the assumptions and predicted orderings hold in the evaluated setting.

## Likely reaction from the same reviewers

### Reviewer 2x5B

The paired MoCo v3 and SimCLR ViT-S experiments remove the objective-backbone confound. The matched FLUX result removes the generator-budget confound. Table 3 adds full fine-tuning and segmentation. The direct anchor-versus-control values now appear in the main text rather than only as a description. This addresses the item the reviewer explicitly said would move the Quality score. A remaining risk is that learned-space local geometry is mixed for SimCLR. The paper now states the narrower fixed-space conclusion and does not claim that global rank must increase.

### Reviewer QPxu

The revision no longer relies on the claim that a particular combination of known components is itself the learning principle. The theoretical framework defines recoverability-weighted support allocation, proves an interior optimum under explicit conditions, and explains why repeated allocation can differ from a frozen policy. Table 5 shows that the exact lower percentile is not critical and that non-extreme sparse acquisition is much stronger than uniform and top-tail acquisition over five cycles. The paired ViT-S results and uncertainty estimates improve the stronger-learner evidence. The matched policy experiment reports each downstream dataset directly. Adaptive BRIDGE leads on Aircraft and Flowers, while one-shot repair leads on Cars and text-to-image generation leads on ImageNet-100-LT. The paper therefore presents the control as evidence about transfer-specific effects rather than a universal policy ranking.

### Reviewer SJHM

The main text now covers the missing literature families and the appendix gives a direct comparison matrix. Claims about SD3 superiority and universal TS synergy are removed. AIDE-style VLM acquisition, sparse-region text-to-image generation, captioned image-to-image, and conventional augmentation are complete under one matched protocol. Their individual downstream means and standard deviations now appear in the main paper without an aggregate ranking.

## Remaining experiments that affect acceptance

1. A consolidated selector robustness table across distance metric, k, percentile range, candidate multiplier, and FPS.
2. Quantile-wise repairability diagnostics if the current fidelity metrics can be validated. Unreliable automatic fidelity scores should not be used.
3. A measured 100k scaling experiment.

Measured 100k scaling and a domain-adapted LoRA generator are useful secondary additions. They should not displace the matched policy and generation controls above.
