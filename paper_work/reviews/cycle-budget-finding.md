# The matched-budget cycle family does not reproduce the five-cycle optimum

`jobs/rebuttal/fixed_cycles.sh` holds total optimizer updates at 9,700 and
total added images at 2,500, and varies how many repair stages that budget is
spread across. Seven-dataset means, three seeds:

| Condition | Linear | kNN |
|---|---:|---:|
| 3 cycles, BRIDGE | **21.54** | **19.66** |
| 5 cycles, BRIDGE | 19.83 | 17.36 |
| 5 cycles, random-add | 18.98 | 17.45 |
| 5 cycles, no-repair | 19.43 | 17.36 |
| 10 cycles, BRIDGE | 18.02 | 13.69 |
| 10 cycles, random-add | 18.45 | 13.61 |

Two things stand out, and both need care before they go near the paper.

## More cycles is monotonically worse here

Under a fixed update and image budget, three cycles beats five, which beats
ten, on both protocols. The manuscript's mechanism table reports the opposite
shape, with five cycles best and both two and ten substantially worse.

The two are not directly comparable: this family trains **from scratch** --
`fixed_cycles.sh` passes no `checkpoint=` argument -- on 9,700 total updates,
while the main results continue from a pretrained source checkpoint. That is
why the absolute numbers sit near 18--21 rather than 42. The disagreement is
therefore between protocols, not a contradiction within one, but the paper
should not cite a five-cycle optimum as if both protocols supported it.

## The arms are separated at c5 by less than seed noise

At five cycles the three arms span 0.85 points under linear probe and 0.09
under kNN. Reviewer QPxu's objection was precisely that re-population performs
close to generative repair; on this family, it does.

## The random-add arm confounds two changes

`random_add` sets `selection=random` **and** keeps
`generator=strong_augmentation`, while `bridge` sets
`generator=stable_diffusion_3`. The two arms therefore differ in both the
acquisition rule and the generator, so the comparison does not isolate whether
generation matters at matched volume. `no_repair` additionally sets
`ood_augmentation=false`, so it adds no images at all and is matched on
updates but not on data volume.

A clean test of "is it just more data" needs a fourth arm: random selection
with the same SD3 generator, at the same volume. That arm does not currently
exist.
