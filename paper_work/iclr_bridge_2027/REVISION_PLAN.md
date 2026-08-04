# ICLR 2027 revision plan

## Claims to retain

- BRIDGE is a label-free feedback framework in which the current SSL representation allocates a finite data-repair budget and is then changed by that intervention.
- Targeted repeated addition improves transfer in the source regimes tested in the paper.
- A sparse but non-extreme operating band is a robust repeated-cycle choice in the submitted five-cycle experiment.

## Claims to remove or narrow

- Do not claim that SimCLR is intrinsically the strongest SSL objective. Paired ViT-S results support SimCLR and MoCo v3, while DINO is inconclusive or negative.
- Do not claim that SD3 is intrinsically stronger than FLUX. Their matched four-task averages are 19.70 and 19.68.
- Do not claim general BRIDGE and TS synergy. The interaction is target dependent and BRIDGE alone is usually stronger.
- Do not call 75--99 optimal. It is a default inside a robust policy family.
- Do not call oracle real restoration duplication or ordinary re-population.
- Do not claim that global effective rank must increase. The defensible geometric claim is local support change near treated anchors relative to matched untreated regions.

## Experiment package

| Priority | Experiment | Main question | Status on 4 August 2026 |
|---|---|---|---|
| 1 | Anchor versus matched-control geometry | Do treated local regions change more than equally sparse untreated regions? | Three post-hoc seed analyses running |
| 1 | Adaptive versus frozen versus one-shot repair | Does recomputing the acquisition policy matter under matched updates and repair volume? | Three seeds per arm running |
| 1 | Conventional augmentation control | Is targeted non-generative augmentation sufficient? | Three seeds running |
| 1 | VLM-captioned text-to-image versus SDEdit | Does image conditioning add value beyond generation from a description of the same sparse region? | Smoke test running; full paired suite prepared |
| 1 | VLM rare-concept data engine | Can a frozen VLM replace BRIDGE's learner-driven acquisition and text-to-image replace local editing? | Implemented; launch after the text-to-image smoke test |
| 2 | Full-cycle TADA-style and extreme-tail comparison | How does mode-window acquisition compare with learning-difficulty and edge acquisition? | Existing short/full partial runs require one unified rerun or careful protocol selection |
| 2 | Distance and selector robustness | Are results stable across normalized L2, cosine, multiscale k, percentiles, pool size, and FPS? | Several sweeps complete; consolidate and fill only missing cells |
| 2 | Repairability by score quantile | Does fidelity fall in the extreme tail while marginal support value rises before it? | Existing percentile and fidelity jobs require aggregation and possible completion |
| 2 | Measured source scaling | How do scoring, generation, and training costs change from 10k to 100k? | 25k and 50k complete; 100k failed and should be resubmitted after priority-1 jobs |
| 3 | Domain-adapted LoRA generator | Does repeated use of a source-adapted generator improve on a frozen general generator? | Not started; expensive and secondary after the direct T2I/SDEdit comparison |

## Background coverage required

The related-work section must explicitly cover and distinguish the following families.

- Classical imbalance and oversampling, including SMOTE.
- Rare-region and anomaly generation, especially DOPING.
- Representation-driven pruning, coreset selection, and concept curation.
- Diffusion editing and supervised augmentation, including SDEdit, DA-Fusion, synthetic ImageNet augmentation, scaling studies, and TADA.
- Synthetic representation learning, including GenRep, synthetic ImageNet clones, StableRep, procedural programs, and DiffAug.
- Iterative data engines, particularly AIDE, with industrial data-engine work treated as context rather than an archival algorithmic baseline.
- Generative features and distillation, including DreamTeacher and DIFT.
- Long-tailed SSL objectives and analyses, including SDCLR, Temperature Schedules, and hidden-imbalance work.

## Theoretical scope

The theory should justify the policy class without pretending to derive implementation constants.

1. Adding valid points near an anchor cannot increase its kNN radius.
2. With diminishing returns to support, dense regions have low marginal value.
3. If generator repairability falls in the contaminant tail, recoverability-weighted utility can have an interior maximum.
4. With concave regional utilities, marginal gains change after repair, which motivates adaptive re-estimation over a frozen allocation.

The corresponding experiments must test the assumptions through local treatment effects, quantile-wise fidelity and utility, and adaptive versus frozen acquisition.
