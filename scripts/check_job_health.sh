#!/bin/bash
# Decide whether a run is stuck, using signals that have actually held up.
#
# Four cheap proxies for "alive" have each been wrong at least once here:
#
#   log age          A cycle takes 8-20 hours and a run prints almost nothing
#                    between cycle boundaries, so healthy runs routinely go
#                    silent for longer than a cycle.  It also missed a real
#                    stall that died 400 minutes into a 1000-minute cycle.
#   GPU utilization  With no dataloader workers the input pipeline runs in the
#                    main process, so a healthy run on a fast card sits at 0%
#                    GPU while it waits on single-threaded data loading.
#   per-process CPU  A run that is GPU-bound uses about one core, and a run
#                    saving a 285 MiB checkpoint uses almost none.
#   read bytes       Zero reads looks damning until the process is writing a
#                    checkpoint rather than reading data.  A job cancelled on
#                    that evidence turned out to be two minutes from finishing
#                    its last cycle.
#
# So require every channel to be idle before calling a stall: no CPU, no reads,
# AND no writes.  Report GPU utilization but never decide on it, and attribute
# it per process, because these cards carry co-tenants whose activity is
# indistinguishable from ours at the card level.
#
# Usage: check_job_health.sh <node> [seconds]
set -euo pipefail
node="${1:?usage: check_job_health.sh <node> [seconds]}"
window="${2:-60}"

ssh -o BatchMode=yes "$node" "bash -s" <<REMOTE
set -euo pipefail
win=$window
nvidia-smi --query-gpu=index,uuid,utilization.gpu --format=csv,noheader | tr -d ' ' > /tmp/hc_g.txt
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits | tr -d ' ' > /tmp/hc_a.txt
declare -A cardof utilof
while IFS=, read -r i u util; do cardof[\$u]=\$i; utilof[\$u]=\$util; done < /tmp/hc_g.txt

pids=(); : > /tmp/hc_s.txt
# Enumerating through nvidia-smi alone hides a run that currently holds no GPU
# memory.  These jobs release the card during CPU-bound phases, so the one most
# worth checking is precisely the one that would be missing.  Seed the list with
# our own training processes first, then let the GPU listing fill in card and
# utilization for whichever of them hold memory.
for p in \$(pgrep -u "\$(id -u)" -f 'python -m experiment' 2>/dev/null); do
  cpu=\$(awk '{print \$14+\$15}' /proc/\$p/stat 2>/dev/null || echo 0)
  rd=\$(awk '/^read_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo 0)
  wr=\$(awk '/^write_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo 0)
  echo "\$p - - 0 \$cpu \$rd \$wr" >> /tmp/hc_s.txt
done
while IFS=, read -r p u m; do
  [ "\$m" -gt 5000 ] 2>/dev/null || continue
  # Already seeded above; replace the placeholder row with card and memory.
  if grep -q "^\$p " /tmp/hc_s.txt; then
    sed -i "s|^\$p - - 0 |\$p \${cardof[\$u]} \${utilof[\$u]} \$m |" /tmp/hc_s.txt
    continue
  fi
  cpu=\$(awk '{print \$14+\$15}' /proc/\$p/stat 2>/dev/null || echo 0)
  rd=\$(awk '/^read_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo 0)
  wr=\$(awk '/^write_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo 0)
  echo "\$p \${cardof[\$u]} \${utilof[\$u]} \$m \$cpu \$rd \$wr" >> /tmp/hc_s.txt
done < /tmp/hc_a.txt

sleep \$win

while read -r p card util mem cpu0 rd0 wr0; do
  cpu1=\$(awk '{print \$14+\$15}' /proc/\$p/stat 2>/dev/null || echo \$cpu0)
  rd1=\$(awk '/^read_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo \$rd0)
  wr1=\$(awk '/^write_bytes/{print \$2}' /proc/\$p/io 2>/dev/null || echo \$wr0)
  dc=\$(( cpu1 - cpu0 )); dr=\$(( (rd1 - rd0) / 1048576 )); dw=\$(( (wr1 - wr0) / 1048576 ))
  cores=\$(( dc / (win * 100) )); frac=\$(( (dc * 10 / (win * 100)) % 10 ))
  job=\$( { tr '\0' '\n' < /proc/\$p/environ; } 2>/dev/null | awk -F= '/^SLURM_ARRAY_JOB_ID=/{a=\$2} /^SLURM_ARRAY_TASK_ID=/{t=\$2} END{if(a!="")print a"_"t}' || true)
  verdict=ALIVE
  if [ ! -r /proc/\$p/io ]; then
    verdict="OTHER-USER"
  elif [ "\$dc" -lt \$(( win * 5 )) ] && [ "\$dr" -eq 0 ] && [ "\$dw" -eq 0 ]; then
    verdict="STALLED?"
  fi
  printf '%-9s pid=%-7s card=%s gpu=%4s mem=%6sMiB cpu=%s.%score read=%sMiB write=%sMiB  %s\n' \
    "\${job:-?}" "\$p" "\$card" "\$util" "\$mem" "\$cores" "\$frac" "\$dr" "\$dw" "\$verdict"
done < /tmp/hc_s.txt
REMOTE
