#!/bin/bash
# One-shot health report for the full-metric backfill across both clusters.
#
# Reports queue drain, worker occupancy, spare GPU capacity that could absorb
# more workers, and any failures that need triage.  Intended to be the single
# command a periodic watcher runs.

set -uo pipefail

GRUENAU_QUEUE="${GRUENAU_QUEUE:-/vol/home-vol2/ml/laitenbf/FOMO_runtime/full_metric_queue}"
ALEX_HOST="${ALEX_HOST:-alex}"
ALEX_QUEUE="${ALEX_QUEUE:-/home/atuin/c107fa/c107fa12/FOMO_runtime/full_metric_queue}"

count() { ls "$1" 2>/dev/null | wc -l | tr -d ' '; }

echo "=== $(date --iso-8601=seconds) ==="

echo
echo "-- gruenau queue --"
for state in pending running completed failed; do
    printf '%-10s %s\n' "$state" "$(count "$GRUENAU_QUEUE/$state")"
done

echo
echo "-- gruenau workers --"
squeue -u "$USER" -h -o "%T %R" 2>/dev/null | sort | uniq -c | sed 's/^/  /'
running_workers="$(squeue -u "$USER" -h -t RUNNING -o "%j" 2>/dev/null | grep -c fomo-metrics)"
pending_workers="$(squeue -u "$USER" -h -t PENDING -o "%j" 2>/dev/null | grep -c fomo-metrics)"
echo "  fomo-metrics running=$running_workers pending=$pending_workers"

echo
echo "-- gruenau spare GPU capacity (wbimlgpu, excluding broken prolog nodes) --"
# Free GPUs that no worker of ours is waiting on: room to submit more.
for node in guppi5 guppi6 guppi7 guppi8; do
    state="$(sinfo -h -n "$node" -p wbimlgpu -o "%t" 2>/dev/null | head -1)"
    gres="$(sinfo -h -n "$node" -p wbimlgpu -o "%G" 2>/dev/null | head -1)"
    alloc="$(squeue -h -w "$node" -o "%b" 2>/dev/null | grep -c "gpu")"
    printf '  %-9s state=%-8s gres=%-18s gpu_jobs=%s\n' \
        "$node" "${state:-?}" "${gres:-?}" "$alloc"
done

echo
echo "-- gruenau recent failures --"
fails="$(count "$GRUENAU_QUEUE/failed")"
if [[ "$fails" == "0" ]]; then
    echo "  none"
else
    ls -t "$GRUENAU_QUEUE/failed" 2>/dev/null | head -10 | sed 's/^/  /'
fi

echo
echo "-- alex --"
timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 "$ALEX_HOST" "
    for state in pending running completed failed; do
        printf '  %-10s %s\n' \"\$state\" \"\$(ls $ALEX_QUEUE/\$state 2>/dev/null | wc -l | tr -d ' ')\"
    done
    echo '  workers:'
    squeue -u \$USER -h -o '    %.10i %.14j %.9T %.10M %R' 2>/dev/null | grep fomo || echo '    none'
" 2>&1 || echo "  alex unreachable"
