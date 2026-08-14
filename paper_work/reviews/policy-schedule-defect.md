# The repair-schedule conditions are not distinguishable as run

In every paper-facing experiment, varying `selection_reuse_policy` or
`repair_once` produces a checkpoint bit-identical to the adaptive baseline, so
the adaptive-versus-frozen and repeated-versus-one-shot comparisons compare a
model against itself.

**The parameters themselves work.** A minimal reproduction -- three cycles,
one epoch, twenty anchors, conventional augmentation -- diverges correctly,
with a maximum weight difference of 2.9e-01 between the two policies. An
earlier version of this note claimed the parameters were inert in general;
that was wrong. Selection also demonstrably differs in the real runs: the
repair manifests for adaptive and frozen overlap on only 14 of 500 anchors
from cycle 1 onward.

So the divergence is real at selection time and real at small scale, but is
lost before it reaches the weights in the full configuration. The reproduction
differs from those runs in scale, schedule and repair operator; the operator is
the prime suspect, because Stable Diffusion 3 is the one component that
persists state across cycles through the HDF5 image store. A second
reproduction fixes everything except the generator to test that directly.

This matters because those comparisons are priority-1 in `REVISION_PLAN.md`
and are the evidence cited for Proposition 4 in `REVIEWER_RISK_AUDIT.md`.

## Evidence

Weight-level comparison of `last.ckpt`, seed 0, all 320 tensors.

### `rebuttal_feedback_*` (Alex, `jobs/rebuttal/adaptive_feedback.sh`)

| Comparison | Max abs. weight difference | Verdict |
|---|---|---|
| adaptive vs **static** | 0.000e+00 | identical |
| adaptive vs **one_shot** | 0.000e+00 | identical |
| adaptive vs dense_placebo | 5.259e+00 | differs |
| adaptive vs top_tail | 5.781e+00 | differs |

### `iclr_fullpolicy5e100_*` and `rebuttal_fullpolicy3e60_*` (Gruenau)

| Comparison | Max abs. weight difference | Verdict |
|---|---|---|
| adaptive vs **static** (5-cycle) | 0.000e+00 | identical |
| adaptive vs **static** (3-cycle) | 0.000e+00 | identical |
| adaptive vs one_shot (5-cycle) | 5.220e-02 | differs |
| adaptive vs no_repair (5-cycle) | 8.922e-02 | differs |

The one-shot arm differs in the `fullpolicy` family only because that launcher
also raises `num_generations_per_ood_sample` from 5 to 15 or 25, which changes
the data volume. The schedule parameter itself contributes nothing.

## What this rules out

- **Not the parameters being unimplemented.** The minimal reproduction shows
  them working; see above.
- **Not a general pipeline fault.** Conditions that change *which* samples are
  selected (`dense`, `top`) and how much data is added take effect normally.
  Only the parameters governing *when* selection is recomputed are inert.
- **Not a single-repair-stage artifact.** `adaptive_feedback.sh` runs
  `max_cycles=6`, so cycles 0-4 all augment. With several repair stages, a
  frozen selector and a one-shot schedule must diverge from adaptive, and they
  do not.
- **Not a launcher bug.** `adaptive_feedback.sh` sets
  `selection_reuse_policy=static_first_cycle` and `repair_once=true` correctly
  and passes both as Hydra overrides.

## Where the implementation looks correct but the effect is absent

- `ImbalancedTraining.py:1018-1025` gates augmentation on `repair_once`.
- `ImbalancedTraining.py:1039-1048` substitutes the frozen indices.
- `ImbalancedTraining.py:1082-1087` populates them after the first cycle.
- `ImbalancedTraining.py:857-868` honours the precomputed indices.

Each reads correctly in isolation, so the defect is in how these interact
across cycles, or in whether the overrides reach `self.args` at the point of
use. That needs an owner decision rather than a speculative patch.

## Detection

`scripts/generate_iclr_tables.py` now warns when two table rows are identical
on every dataset. It currently fires on
`Adaptive BRIDGE == Frozen first-cycle selector`. Under kNN the duplicate rows
agree to the last digit; under linear probe they differ only by probe-training
noise on identical features, which is what disguised the problem.

## Until it is resolved

Do not report the frozen first-cycle selector or the one-shot arm of the
feedback family as distinct conditions, and do not cite either as evidence for
Proposition 4.
