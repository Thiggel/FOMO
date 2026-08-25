# RESOLVED: the repair-schedule conditions were not distinguishable as run

**Cause found (commit a29d244).** The schedule parameters were never inert.
Each cycle builds a fresh `L.Trainer`, so `global_step` restarts at zero, and
every cycle stops at the same `max_steps`. The `ModelCheckpoint` callbacks are
reused across those trainers, and Lightning skips a save whenever
`_last_global_step_saved` equals the current `global_step`. Cycle 1 saved at
step 4850; every later cycle also ended at 4850, so its save was treated as a
duplicate and silently dropped. `last.ckpt` therefore held the cycle-1 encoder
for the whole run, and any two arms that share cycle 1 published the same
weights no matter how far their later cycles diverged.

This explains every observation below, including the two that ruled out the
obvious hypotheses: selection genuinely diverged (14/500 anchor overlap from
cycle 1), and the minimal reproduction diverged correctly (it ran few enough
steps that the guard never fired).

The fix resets the callbacks' `_last_global_step_saved` to 0 before each
cycle's trainer is constructed. `scripts/check_duplicate_encoders.py` gates
table generation on encoder fingerprints so this class of fault cannot reach a
table again.

**Scope of the damage.** Only runs with two or more training fits were
affected: `main_generalization_vits_full` and
`main_policy_repair_controls_full`. The selector-robustness and
percentile-utility sweeps run a single fit and are unaffected.

The original diagnosis is kept below for the record.

---

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
