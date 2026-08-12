# The sparsity band barely matters; diversity within it does

The percentile sweep spends an identical acquisition budget on different
quantile ranges of the kNN score distribution, holding the rest of the loop
fixed. Across eight of nine bands and all seven downstream datasets, the band
has almost no effect.

Seven-dataset mean, three seeds per band:

| Band | Linear | kNN |
|---|---:|---:|
| q0--25 (densest) | 27.99 | 24.74 |
| q25--50 | 27.88 | 25.06 |
| q50--75 | 27.17 | 24.88 |
| q75--85 | 26.94 | 24.96 |
| q85--90 | **28.27** | **25.08** |
| q90--95 | 27.88 | 24.94 |
| q97--99 | 27.98 | 25.02 |
| q99--100 (extreme tail) | 27.96 | 24.93 |

The full spread is 1.32 points under linear probe and 0.35 under kNN, against
per-dataset seed deviations of roughly 0.5 to 3 points. The nominal best band
is interior (q85--90), but its margin over the extreme tail is 0.31 and 0.15,
far inside the noise.

Two things follow, and the second is the useful one.

## There is no measurable interior optimum

Proposition 3 predicts that utility peaks at an interior band because repair
fidelity collapses in the extreme tail. These runs do not show that peak, and
they do not show the extreme tail being harmful. The claim the data supports
is the weaker one already made in Table 5: the exact cutoff is not critical.

## The extreme tail is only harmful without diversity

This looks like it contradicts the mechanism table, where top-tail
acquisition is far weaker than the mode window. It does not, because the two
operations differ in more than the band:

- `ood_percentile_bin` selects the band and then applies farthest-point
  sampling inside it (`ood.py:193-203`), including for q99--100.
- `ood_selection_strategy=top` falls through to
  `sorted_indices_desc[:num_samples]` (`ood.py:238-239`) with no diversity
  step at all.

So the arm that performs badly is the one that takes the most extreme samples
*and* accepts near-duplicates, while the extreme band sampled diversely
performs like every other band. On this evidence the penalty attributed to
extreme sparsity is better explained by lost coverage than by unrepairable
anchors — which points at the farthest-point-sampling argument the paper
already makes via Gonzalez, rather than at recoverability weighting.

## Confidence and the experiment that would settle it

The comparison spans protocols: the percentile sweep is a two-cycle run with a
single repair stage, while the mechanism table is the five-cycle schedule, so
the absolute numbers are not comparable. The reconciliation above is therefore
suggestive rather than established.

One cheap experiment settles it: run q99--100 with and without
farthest-point sampling under a single protocol. If the diversity-free variant
collapses and the diverse one does not, the coverage explanation is confirmed
and Proposition 3 should be reframed around diversity rather than fidelity.
