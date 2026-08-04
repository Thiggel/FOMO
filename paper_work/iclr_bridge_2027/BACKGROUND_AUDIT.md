# Background and closest-work audit

This file is the coverage checklist for the ICLR manuscript. It separates historical context from direct algorithmic predecessors and records the comparison needed for every close family.

## Long-tailed learning and self-supervised learning

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| Class reweighting and resampling | Uses known labels or frequencies to change supervised training mass | Brief supervised background and a clear statement that BRIDGE does not observe source labels or frequencies |
| SMOTE | Foundational labeled minority interpolation | Historical citation only |
| Supervised contrastive long-tail methods | Learns balanced supervised feature spaces | Explain why labels and task loss make the setting different |
| SDCLR | Changes contrastive optimization through a self-competitor on long-tailed unlabeled data | Main baseline under matched backbone and training protocol |
| Temperature Schedules | Changes the contrastive optimization trajectory while holding the dataset fixed | Main complementary baseline and interaction ablation |
| Hidden class imbalance analyses | Establish that SSL is sensitive to latent source imbalance | Motivation and positioning |

## Density, rarity, selection, and curation

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| Deep kNN OOD detection | Supplies the nonparametric local-distance primitive | Cite as an established primitive, with no novelty claim |
| DOPING | Oversamples infrequent normal examples near the edge of a learned latent distribution for anomaly detection | Prominent comparison; use an extreme-score acquisition control and distinguish anomaly detection from transfer-oriented cyclic SSL |
| Coresets and farthest-first traversal | Allocate a finite subset budget by representation-space coverage | Cite FPS as an established approximation and ablate it against random choice in the same candidate set |
| Dataset pruning and scaling-law pruning | Removes redundant examples to improve training efficiency | Contrast removal with support addition |
| Concept-cluster pruning and embedding curation | Uses clusters or concepts to curate model-training data | Include an inverse-cluster-frequency acquisition baseline |
| Active learning and hard-example mining | Uses model state to acquire labels or emphasize difficult examples | Explain that BRIDGE acquires unlabeled inputs and never queries annotations |

## Diffusion editing and supervised augmentation

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| SDEdit | Direct ancestry of add-noise-and-denoise image editing | Cite in the method and compare image-conditioned editing with generation from noise |
| DA-Fusion | Off-the-shelf diffusion editing for few-shot supervised recognition, with domain adaptation through textual inversion | Prominent related work; distinguish labels and one-shot augmentation; discuss LoRA or textual-inversion domain adaptation |
| Real Guidance and related faithful editing | Image-conditioned generative augmentation with fidelity controls | Cover alongside DA-Fusion and compare fidelity/diversity metrics |
| Synthetic ImageNet augmentation by Azizi et al. | Large class-conditional diffusion augmentation improves supervised recognition | Use as evidence that external generator priors and generated volume need controls |
| Scaling laws of synthetic images by Fan et al. | Shows that prompts, guidance, generator, diversity, and source-domain alignment matter | Motivate complete generator-setting reporting and matched FLUX/SD3 comparison |
| TADA | Selects slow-learned supervised examples and generates faithful variants, with a theoretical feature-learning analysis | Closest targeted augmentation baseline; adapt its loss/learning-speed signal to SSL under the same generator, budget, and cycles |
| DIAGen, Diff-Mix, DPT, and other prompt/diversity methods | Improve semantic variation, interpolation, or prompt diversity in supervised generative augmentation | Cover as broader alternatives; one prompt-based control is required, not a separate full reproduction of every method |

## Synthetic data for representation learning

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| Ren and Lee synthetic representation learning | Early synthetic imagery and self-supervised objectives for transfer | Historical citation |
| Learning to See by Looking at Noise and procedural image programs | Learns visual representations from designed synthetic processes and emphasizes diversity | Explain the difference between a wholly synthetic source and local repair of an observed corpus |
| GenRep | Learns contrastive representations from a black-box generator and latent-neighborhood views | Prominent comparison in the synthetic-SSL section |
| Fake It Till You Make It | Trains transferable models on synthetic ImageNet clones | Contrast global class-prompt synthesis with label-free local source intervention |
| StableRep | Learns representations from text-to-image samples and uses same-prompt generations as positives | Prominent comparison; discuss generator configuration sensitivity |
| DiffAug | Iteratively trains a semantic encoder and conditional diffusion generator to produce contrastive positives | Closest iterative unsupervised generative-learning predecessor; distinguish learned generic positive generation from frozen budgeted support allocation |

## Iterative data engines and language-mediated generation

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| AIDE | Uses VLMs and LLMs to identify failures, curate and auto-label driving data, and verify an open-world detector iteratively | Prominent systems and algorithm comparison; include a label-free VLM rare-concept and text-to-image adaptation |
| Omniverse Replicator | Industrial synthetic-data infrastructure and iterative model-data workflow | Systems context only |
| Autonomous-driving data-engine keynotes and industrial practice | Motivates cyclic model-data improvement | Non-archival context or footnote, not a scientific novelty baseline |
| VLM-guided prompt generation | Converts visual or task failures into language before synthesis | Direct control against image-conditioned SDEdit |
| Domain-specific LoRA or textual inversion | Adapts the generator to source concepts before repeated generation | Secondary generator control after the direct from-noise comparison |

## Diffusion models as external teachers

| Work or family | Relation to BRIDGE | Required treatment |
|---|---|---|
| DreamTeacher | Distills generative features into a discriminative backbone | Direct feature-distillation control |
| DIFT | Demonstrates strong semantic information in diffusion activations | Justifies extracting a frozen diffusion teacher representation |
| REPA | Aligns diffusion-model representations to an external discriminative encoder | Clarify that its transfer direction is the reverse of generator-to-SSL distillation |
| Diffusion feature extraction and correspondence methods | Show that frozen diffusion models contain useful perception features | Broader context for the external-prior limitation |

## Exact novelty position

BRIDGE does not claim novelty for kNN distance, FPS, diffusion editing, synthetic SSL, targeted supervised augmentation, or the general idea of an iterative data engine. The contribution is the joint operating regime in which all of the following hold.

1. No source labels, class frequencies, downstream labels, or supervised task failures are used.
2. Acquisition is driven by local support in the current SSL representation.
3. A finite budget adds new local data rather than only reweighting or removing existing samples.
4. The acquisition signal is recomputed after the intervention changes the representation.
5. The paper isolates these properties through matched one-shot, frozen, adaptive, uniform, extreme-tail, conventional-augmentation, real-restoration, VLM-guided, and distillation controls.

The ICLR paper should claim a new problem formulation and an empirically characterized feedback policy. It should not claim that no previous work has coupled learning and generation iteratively.
