#!/bin/bash
# Emit one line per task that leaves the queue, classified by how it left.
# Silence means every task is still running; the watch ends when the queue drains.
cd "${FOMO_REPO_DIR:-/vol/home-vol2/ml/laitenbf/FOMO}"
poll="${WATCH_POLL_SECONDS:-300}"
beat="${WATCH_HEARTBEAT_POLLS:-48}"      # 48 * 5 min = 4 h
# Record each task's name while it is still in the queue.  Reading the name
# back from slurm-<id>.out afterwards works only for jobs that write their
# log here, so another project's array under the same account was reported as
# a task that never started.
declare -A jobname
note_names() {
  local id name
  while read -r id name; do
    [[ -n "$id" ]] && jobname[$id]="$name"
  done < <(squeue -u "$USER" -h -r -o '%i %j' 2>/dev/null)
}
note_names
prev="$(squeue -u "$USER" -h -r -o '%i' 2>/dev/null | sort)"
n=0
while :; do
  sleep "$poll"
  cur="$(squeue -u "$USER" -h -r -o '%i' 2>/dev/null | sort)"
  # A squeue that errors returns empty; do not read that as "everything finished".
  if [[ -z "$cur" ]] && ! squeue -u "$USER" -h -r >/dev/null 2>&1; then continue; fi
  while read -r id; do
    [[ -z "$id" ]] && continue
    log="slurm-${id}.out"
    sig="$(grep -hoE 'unable to open shared memory|torch.OutOfMemoryError|CUDA error|Giving up: GPU|DataLoader worker.*(killed|exited)|Missing common source|CANCELLED|DUE TO TIME LIMIT|terminate called without an active exception|Cannot allocate memory|oom-kill' "$log" 2>/dev/null | sort -u | tr '\n' ' ')"
    name="$(grep -hoE 'experiment_name=[A-Za-z0-9_]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2)"
    name="${name:-${jobname[$id]:-}}"
    # A task cancelled before it ever ran leaves no output file at all.  That
    # is a deliberate act, not a fault, and reporting it as an unexplained
    # disappearance buries the real failures in noise over a multi-day run.
    # Another project shares this account and resubmits every few minutes.
    # A task of ours is named fomo-* and writes its log here, so a task with
    # neither is not ours and reporting it only buries what is.
    if [[ ! -e "$log" && "${jobname[$id]:-}" != fomo-* ]]; then
      continue
    fi
    if [[ ! -e "$log" ]]; then
      echo "GONE    $id  ${name:-?}  -> left the queue with no log here"
    elif [[ -n "$sig" ]]; then
      echo "FAILED  $id  ${name:-?}  -> $sig"
    elif grep -q 'print_mean_std' "$log" 2>/dev/null; then
      echo "DONE    $id  ${name:-?}"
    else
      echo "GONE    $id  ${name:-?}  -> no result and no known error; check $log"
    fi
  done < <(comm -23 <(echo "$prev") <(echo "$cur"))
  note_names
  prev="$cur"
  n=$((n+1))
  if [[ -z "$cur" ]]; then echo "ALL DONE: queue empty at $(date '+%F %H:%M')"; exit 0; fi
  if (( n % beat == 0 )); then
    echo "still running: $(squeue -u "$USER" -h -r -t RUNNING | wc -l) running, $(squeue -u "$USER" -h -r -t PENDING | wc -l) pending ($(date '+%F %H:%M'))"
  fi
done
