#!/bin/bash
# Reclaim disk from artefacts nothing reads any more.
#
# Prints what it would remove and exits; pass --apply to actually delete.
#
# Deliberately NOT touched:
#   checkpoints/<paper experiments>  74 queued cells read last.ckpt from here
#   dataset_downloads                30 queued cells need the PASS archive
#   hf, data, voc2012                read by the running benchmarks
#   **/repair_manifests/             the pending repair-fidelity analysis
#   rebuttal_runs/synthetic_provenance   pending provenance analysis
#   rebuttal_runs/percentile_utility     a training job is reading it now

set -uo pipefail

RUNTIME="${BASE_CACHE_DIR:-/vol/home-vol2/ml/laitenbf/FOMO_runtime}"
apply=""
[[ "${1:-}" == "--apply" ]] && apply=1

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0}"; }

total=0

# Sum the matched files themselves.
report_files() {
    local label="$1"; shift
    local bytes count
    bytes="$("$@" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    count="$("$@" -printf '.\n' 2>/dev/null | wc -l)"
    printf '%-42s %10s  %s files\n' "$label" "$(human "$bytes")" "$count"
    total=$(( total + bytes ))
}

# Sum what the matched directories *contain*.  find -printf '%s' on a
# directory reports its inode size, a few KB, not the tree beneath it, so
# using report_files here would understate this category by ~29 GB.
report_dirs() {
    local label="$1"; shift
    local bytes count
    bytes="$("$@" -print0 2>/dev/null | du -sc --files0-from=- --block-size=1 2>/dev/null |
        tail -1 | cut -f1)"
    count="$("$@" -printf '.\n' 2>/dev/null | wc -l)"
    printf '%-42s %10s  %s dirs\n' "$label" "$(human "${bytes:-0}")" "$count"
    total=$(( total + ${bytes:-0} ))
}

# 1. Benchmark checkpoints Lightning wrote and nothing reads.  The leak itself
#    is fixed (enable_checkpointing=False), so this is a one-off backlog.
lightning_ckpt=(find "$RUNTIME/lightning/lightning_logs" -type f -name '*.ckpt')

# 2. Generated image shards.  Regenerating needs GPU time, so this is the one
#    category worth pausing over -- but no queued cell reads them, and the
#    manifests recording what was selected and generated are kept.
generated_h5=(find "$RUNTIME/rebuttal_runs" -type f -name '*.h5'
    -not -path '*/synthetic_provenance/*'
    -not -path '*/percentile_utility/*')

# 3. Superseded checkpoint families: pilots, the screening wave, and the
#    deadline runs that rebuttal_full5e100_* replaced.
superseded=(find "$RUNTIME/checkpoints" -mindepth 1 -maxdepth 1 -type d
    \( -name 'smoke_*' -o -name 'screen_*' -o -name 'deadline_*' \))

echo "=== reclaimable ==="
report_files "lightning benchmark checkpoints" "${lightning_ckpt[@]}"
report_files "generated image shards (.h5)" "${generated_h5[@]}"
report_dirs "superseded checkpoint families" "${superseded[@]}"
echo "---"
printf '%-42s %10s\n' "total" "$(human "$total")"

if [[ -z "$apply" ]]; then
    echo
    echo "Dry run. Re-run with --apply to delete."
    exit 0
fi

echo
echo "Deleting..."
"${lightning_ckpt[@]}" -delete
"${generated_h5[@]}" -delete
"${superseded[@]}" -exec rm -rf {} +
echo "Done. Now: $(df -h "$RUNTIME" | tail -1)"
