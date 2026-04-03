# Running Paper Experiments

Updated: 2026-03-21 Europe/Berlin

## Current rerun wave

These are the only reruns that were launched after patching the concrete failure causes.

### Submitted with fixes applied

- `3471828` `jobs/ablations/pretraining/moco.sh`
  - Array: `0`
  - Why: previous seed `0` hit an `OUT_OF_MEMORY` kill while seeds `1` and `2` completed.
  - Fix coverage: reduced dataloader worker pressure via `FOMO_NUM_WORKERS=2`.

- `3471829` `jobs/ablations/generation/flux.sh`
  - Array: `0-2`
  - Why: all three previous seeds timed out.
  - Fix coverage: FLUX batching is now actually wired up in code and the job uses `flux_batch_size=2`; rerun also uses `FOMO_NUM_WORKERS=2`.

- `3471830` `jobs/ablations/architecture/vit_s.sh`
  - Array: `0-2`
  - Why: all previous seeds failed immediately with DDP unused-parameter errors.
  - Fix coverage: ViT models now use `ddp_find_unused_parameters_true`; rerun also uses `FOMO_NUM_WORKERS=2`.

- `3471831` `jobs/ablations/architecture/vit_b.sh`
  - Array: `0-2`
  - Why: all previous seeds failed immediately with DDP unused-parameter errors.
  - Fix coverage: ViT models now use `ddp_find_unused_parameters_true`; rerun also uses `FOMO_NUM_WORKERS=2`.

- `3471832` `jobs/sota/cifar-100-lt/bridge.sh`
  - Array: `2`
  - Why: previous seed `2` failed with transient HDF5 read errors followed by NCCL watchdog abort.
  - Fix coverage: HDF5 reads now retry on transient busy/open-file errors; rerun also uses `FOMO_NUM_WORKERS=2`.

- `3471833` `jobs/sota/cifar-100-lt/bridge_ts.sh`
  - Array: `1`
  - Why: previous seed `1` failed with `Too many open files` while reading generated-image HDF5 storage.
  - Fix coverage: HDF5 reads now retry on transient busy/open-file errors; rerun also uses `FOMO_NUM_WORKERS=2`.

- `3471834` `jobs/sota/pass-subset/bridge_ts.sh`
  - Array: `0-1`
  - Why: previous seeds `0` and `1` failed with transient HDF5 read errors followed by NCCL watchdog abort.
  - Fix coverage: HDF5 reads now retry on transient busy/open-file errors; rerun also uses `FOMO_NUM_WORKERS=2`.

## Intentionally not resubmitted yet

- `jobs/ablations/cycles/cycles_10.sh`
  - Reason: the failed runs only reached cycle `8/10` by roughly `20.5h`, so the logged pace still does not fit the `24h` walltime even if the file-handle errors are removed.

- `jobs/ablations/cycles/cycles_20.sh`
  - Reason: the timed-out runs reached only cycle `12/20` or `13/20` by `24h`, so this configuration still cannot complete within the current Slurm walltime.

## Patches backing these reruns

- `experiment/models/finetuning_benchmarks/BaseKNNClassifier.py`
  - Fixed the worker-count bug so `FOMO_NUM_WORKERS` actually affects `$k$NN extraction.

- `experiment/dataset/ImageStorage.py`
  - Added retry logic for transient HDF5 read failures with errno `16` and `23`.

- `experiment/__main__.py`
  - Use `ddp_find_unused_parameters_true` for ViT models.

- `experiment/ImbalancedTraining.py`
  - FLUX generation now batches multiple source images instead of always calling the pipeline one image at a time.

- `jobs/ablations/generation/flux.sh`
  - Explicitly sets `flux_batch_size=2` for the rerun.

## Chained cycle-ablation reruns

These use explicit cycle segmentation plus Slurm `afterok` dependencies so later segments start only after the earlier checkpoint exists.

- `3471855` `ablations_cycles_10-seg1`
  - Array: `0-2`
  - Cycles: `0` to `5`
  - Finetune: `false`
- `3471856` `ablations_cycles_10-seg2`
  - Array: `0-2`
  - Dependency: `afterok:3471855`
  - Cycles: `5` to `10`
  - Finetune: `true`

- `3471857` `ablations_cycles_20-seg1`
  - Array: `0-2`
  - Cycles: `0` to `7`
  - Finetune: `false`
- `3471858` `ablations_cycles_20-seg2`
  - Array: `0-2`
  - Dependency: `afterok:3471857`
  - Cycles: `7` to `14`
  - Finetune: `false`
- `3471859` `ablations_cycles_20-seg3`
  - Array: `0-2`
  - Dependency: `afterok:3471858`
  - Cycles: `14` to `20`
  - Finetune: `true`

## Retries after recent failures

- `3486885` `jobs/ablations/generation/flux.sh` (array `0-2`, `FOMO_NUM_WORKERS=2`)
- `3486886` `jobs/ablations/pretraining/moco.sh` (array `0`)
- `3486887` `jobs/ablations/architecture/vit_s.sh` (array `0-2`)
- `3486888` `jobs/ablations/architecture/vit_b.sh` (array `0-2`)
- `3486889` `jobs/sota/cifar-100-lt/bridge.sh` (array `2`)
- `3486890` `jobs/sota/cifar-100-lt/bridge_ts.sh` (array `1`)
- `3486891` `jobs/sota/pass-subset/bridge_ts.sh` (array `0-1`)

## Fresh segmented reruns submitted on 2026-03-23

- `3489461` `ablations_cycles_10-seg1`
- `3489462` `ablations_cycles_10-seg2` depends on `3489461`
- `3489463` `ablations_cycles_20-seg1`
- `3489464` `ablations_cycles_20-seg2` depends on `3489463`
- `3489465` `ablations_cycles_20-seg3` depends on `3489464`
- `3489466` `ablations-pretraining-moco-seg1`
- `3489467` `ablations-pretraining-moco-seg2` depends on `3489466`
- `3489468` `ablations-architecture-vit-s-seg1`
- `3489469` `ablations-architecture-vit-s-seg2` depends on `3489468`
- `3489470` `ablations-architecture-vit-b-seg1`
- `3489471` `ablations-architecture-vit-b-seg2` depends on `3489470`
- `3489472` `ablations-generation-flux-seg1`
- `3489473` `ablations-generation-flux-seg2` depends on `3489472`
- `3489474` `ablations-generation-flux-seg3` depends on `3489473`
- `3489475` `sota-cifar-100-lt-bridge-seg1`
- `3489476` `sota-cifar-100-lt-bridge-seg2` depends on `3489475`
- `3489477` `sota-cifar-100-lt-bridge-ts-seg1`
- `3489478` `sota-cifar-100-lt-bridge-ts-seg2` depends on `3489477`
- `3489479` `sota-pass-subset-bridge-ts-seg1`
- `3489480` `sota-pass-subset-bridge-ts-seg2` depends on `3489479`

## 2026-03-26 resume retry wave
- Persistent-worker teardown crash on resumed ViT/FLUX segments fixed by disabling persistent workers in `experiment/dataset/ImbalancedDataModule.py`.
- Replaced array-level continuation chains with per-seed chains so successful seeds are no longer blocked by array-wide `afterok` dependencies.
- New job chains:
  - ViT-S: 3496701-3496708
  - ViT-B: 3496709-3496715
  - FLUX: 3496716-3496725

## 2026-03-30 remaining cleanups
- Queue was cleaned of stale `DependencyNeverSatisfied` jobs for PASS BRIDGE+TS and `cycles-20`.
- PASS BRIDGE+TS resume issue identified: the checkpoints live under `experiment_dataset_hf_scripts_pass_subset.py`, not `mlfoundations_PASS`.
- Added `CHECKPOINT_DATASET_ID` support to `jobs/segment_resume_generic.sh` so future segmented resumes can target a different checkpoint subdirectory than the logical dataset identifier.
- Resubmitted only the remaining unfinished jobs:
  - PASS BRIDGE+TS seed `0`: `3504094 -> 3504095`
  - PASS BRIDGE+TS seed `1`: `3504096 -> 3504097 -> 3504098`
  - cycles-20 seed `0`: `3504099 -> 3504100 -> 3504101 -> 3504102 -> 3504103 -> 3504104`
  - cycles-20 seed `2`: `3504105 -> 3504106 -> 3504107 -> 3504108 -> 3504109 -> 3504110`
  - cycles-20 seed `1`: `3504111 -> 3504112 -> 3504113 -> 3504114 -> 3504115 -> 3504116 -> 3504117 -> 3504118 -> 3504119`

## 2026-04-01 final remaining retries
- Cancelled dead dependency jobs:
  - `3504457`, `3504460`, `3505607`, `3505608`
- Resubmitted PASS BRIDGE+TS with finer remaining split:
  - seed `0`: `3->4`, `4->5`, finetune
  - seed `1`: `3->4`, `4->5`, finetune
- Resubmitted `cycles-20` seed `0` from `17->19`, `19->20`, finetune.
- Excluded node `a0633` for the `cycles-20` seed `0` retry because the previous `17->19` failure was an NCCL watchdog abort on that node, not a walltime or quota failure.
- 2026-04-01: cancelled dead cycles-20 seed-0 continuation jobs `3506853`, `3506854`.
- Moved corrupted cycle-17 cache aside: `/home/atuin/c107fa/c107fa12/FOMO2/ablations_cycles_20_seed_0/17_corrupt_20260401_140710`.
- Resubmitted cycles-20 seed-0 as `3507695` -> `3507696` -> `3507697`.
- 2026-04-01: cycles-20 seed-0 r7 failed due to missing HDF5 shard in cycle 17.
- Cancelled dead jobs: 3507696, 3507697.
- Moved missing cycle-17 dir aside: /home/atuin/c107fa/c107fa12/FOMO2/ablations_cycles_20_seed_0/17_missing_20260401_150657
- Resubmitted seed-0 with regeneration step: 16->17 -> 17->19 -> 19->20 -> finetune (jobs 3507801 -> 3507802 -> 3507803 -> 3507804).
- 2026-04-02: cycles-20 seed-0 r8 failed due to stale cycle-17 entry in image_counts.
- Removed cycle 17 from /home/atuin/c107fa/c107fa12/FOMO2/ablations_cycles_20_seed_0_image_counts.pkl.
- Ensured cycle-17 dir absent to regenerate.
- Resubmitted seed-0 as r9: 3510295 -> 3510296 -> 3510297 -> 3510298.
