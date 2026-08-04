# Rebuttal protocol audit

## Blocking table discrepancy

The submitted manuscript contains two incompatible ImageNet-100-LT SimCLR
comparisons:

| Table family | SimCLR | BRIDGE | Average tasks |
|---|---:|---:|---|
| Main baseline / ablation | 31.2 | 42.3 | 7 |
| SOTA source-regime table | 39.8 | 42.5 | 5 in main, 7 in appendix |

The corresponding launch scripts do not explain the SimCLR discrepancy:
`jobs/baseline/imbalanced.sh` and `jobs/sota/imagenet-100-lt/simclr.sh` both use
ResNet-50, SimCLR, ImageNet-100-LT, five cycles, 100 epochs per cycle, batch
size 512, and no OOD augmentation. The two values therefore must not be
presented as distinct protocols without locating their original logs and code
revisions.

## Canonical rebuttal protocol

Use the new common-checkpoint causal protocol for every rebuttal comparison:

1. Source-only SimCLR checkpoint: 4,850 optimizer steps.
2. All branches start from the same paired-seed checkpoint.
3. Continued no-repair and repair branches each receive another 4,850 steps.
4. Repair branches add exactly 500 anchors × 5 variants.
5. Report no-repair, uniform, BRIDGE-Lite, conventional augmentation, and SD3
   from the same causal factorial rather than mixing historic table rows.

The canonical source checkpoints are stored under
`rebuttal_branch_source/clane9_imagenet-100/seed_{0,1,2}`. The canonical full
BRIDGE checkpoints are stored under
`rebuttal_factorial_mode_sd3/clane9_imagenet-100/seed_{0,1,2}`.

## Required paper action

- Replace both historic SimCLR/BRIDGE rows with the canonical paired results,
  or retain one historic table only after recovering and documenting its exact
  logs, commit, update count, and downstream suite.
- Never compare the 4,850-step branch checkpoint directly against the
  9,700-step repair branch; use the 9,700-step continued no-repair arm.
- Report optimizer steps, source exposures, generated-image count, final
  cardinality, and GPU-hours directly in each relevant table caption.
