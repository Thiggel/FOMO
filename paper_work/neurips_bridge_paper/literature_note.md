# Literature Note (2026-03-16)

## Search scope
I searched primary sources on 2026-03-16 across arXiv, OpenReview, CVF Open Access, and publisher pages for combinations of:
- `long-tail self-supervised learning`
- `imbalanced self-supervised contrastive learning`
- `OOD long-tail SSL`
- `diffusion long-tail recognition`
- `self-supervised long-tail 2025 2026`

## Methods used in the paper
These are the main directly relevant prior methods that the paper now discusses.
- TS: https://arxiv.org/abs/2303.13664
- SDCLR: https://arxiv.org/abs/2106.02990
- COLT: https://openreview.net/forum?id=v8JIQdiN9Sh
- FASSL: https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/papers/Lin_Frequency-Aware_Self-Supervised_Long-Tailed_Learning_ICCVW_2023_paper.pdf

## Newer work since the project started
I found one clearly relevant new method in the same broad regime that post-dates the earlier draft:
- Hoang, Lee, Kang. "Unsupervised contrastive learning using out-of-distribution data for long-tailed dataset." Neurocomputing 649, published October 7, 2025.
  - DOI / publisher page: https://doi.org/10.1016/j.neucom.2025.130779
  - arXiv preprint: https://arxiv.org/abs/2506.12698

## Assessment
- This 2025 method is relevant enough to be aware of.
- It is not a clean drop-in comparison to BRIDGE because it relies on an external OOD corpus (the paper reports using a 300K random-image OOD set) rather than repairing the source set with generation.
- I did not find an additional 2026 paper that is as directly comparable for computer-vision SSL on long-tailed unlabeled data.
- Because you did not benchmark against the 2025 method, I kept it out of the paper body and recorded it here instead.

## Practical conclusion
- Yes: at least one new adjacent/direct competing method appeared after the original draft phase.
- No: I did not find a second post-2025 method that is as directly comparable as TS / SDCLR / COLT in your exact SSL + long-tail + unlabeled-image setting.
