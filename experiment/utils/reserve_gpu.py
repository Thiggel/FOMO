"""Hold the card's spare memory so a co-tenant cannot take it mid-run.

These nodes let several processes share one GPU, and not all of them come
through Slurm: a card that was empty when the guard checked it can pick up
neighbours hours later.  DINO bridge seed 2 died that way at cycle 3 of 5 after
44 hours, and SimCLR bridge seed 2 at cycle 4 of 5 after 25.  Checking free
memory at startup cannot prevent either, because the memory is taken while the
run is already going.

Allocating a block and freeing it is not enough.  The pages go back to torch's
caching allocator, and this codebase calls ``torch.cuda.empty_cache`` at ten
points, mostly to make room for the diffusion model between cycles.  Each of
those hands the reservation back to the driver, which is why a run that
reserved 34 GiB was later seen holding 5.7 GiB with 39.7 GiB free beside it.

So the block is kept alive for the life of the process.  It is sized to leave
the run the peak it actually needs and to make the rest of the card
unavailable, so a later arrival finds no room instead of taking memory this run
is going to want.  FOMO_GPU_PEAK_MIB is what to leave free; unset means off.
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
