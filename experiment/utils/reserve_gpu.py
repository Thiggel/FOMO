"""Hold the card's spare memory so a co-tenant cannot take it mid-run.

Off unless FOMO_GPU_PEAK_MIB is set, and no launcher sets it.  Read the whole
of this before turning it on.

The problem is real: several processes share one GPU here, not all of them
arrive through Slurm, and a card that was empty when the startup guard checked
it can pick up neighbours hours later.  Two long objective runs were lost that
way, one at cycle 3 of 5 after 44 hours and one at cycle 4 of 5 after 25.

Holding the spare memory does prevent that, and it also caused seven of eight
policy runs to die of CUDA OOM inside an hour.  Those eight were the only jobs
on a ten-card node, so each one claimed the spare on its own card, and the sum
of what they held plus what they needed exceeded the node.  The reservation
turns a card another process might have taken into a card this process has
certainly taken, which is an improvement only while the peak passed in here is
larger than the run's true peak.  The policy arms were given 18000 and grew to
19.6 GiB, so the reservation was the difference between a run that fits and one
that does not.

Use it for a single long run on a contested card, with a peak measured from a
healthy run of that same arm and rounded up.  Do not use it for an array whose
tasks land on one node together.
"""

import os

import torch

_ballast = None


def reserve_gpu_memory() -> None:
    global _ballast
    peak = int(os.environ.get("FOMO_GPU_PEAK_MIB", "0"))
    if peak <= 0 or not torch.cuda.is_available():
        return
    free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    # Leave the run its peak, plus a margin for the fluctuation around it, and
    # the driver its own working room.
    ballast = free_mib - peak - 2048
    if ballast <= 0:
        print(
            f"No spare memory to hold: {free_mib} MiB free, run needs {peak} MiB"
        )
        return
    try:
        _ballast = torch.empty(
            ballast * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
    except torch.OutOfMemoryError:
        print(f"Could not hold {ballast} MiB; continuing without it")
        return
    print(
        f"Holding {ballast} MiB of spare GPU memory; "
        f"{peak} MiB left for the run"
    )
