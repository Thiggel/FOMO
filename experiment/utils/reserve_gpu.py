"""Claim the card's memory before a co-tenant does.

These nodes let several processes share one GPU, and not all of them come
through Slurm: a card that was empty when the guard checked it can pick up two
neighbours hours later.  DINO bridge seed 2 died that way at cycle 3 of 5 after
44 hours, holding 13.6 GiB while two arrivals held 21.3 and 12.5 GiB of the
same 47.4 GiB card.  Waiting for free memory at startup cannot prevent this,
because the memory is taken while the run is already going.

Allocating a block and freeing it leaves the pages with torch's caching
allocator, which does not hand them back to the driver.  The run then grows
into its own cache instead of competing for what is left, and a later arrival
sees the card as full rather than taking what this run will need.

Set FOMO_GPU_RESERVE_MIB to the peak the run needs.  Off when unset.
"""

import os

import torch


def reserve_gpu_memory() -> None:
    want = int(os.environ.get("FOMO_GPU_RESERVE_MIB", "0"))
    if want <= 0 or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    # Leave the driver its own working room, and never fail the run over this:
    # a smaller reservation is still worth having.
    want = min(want, free // (1024 * 1024) - 1024)
    if want <= 0:
        return
    try:
        block = torch.empty(want * 1024 * 1024, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        print(f"GPU reservation of {want} MiB did not fit; continuing without it")
        return
    del block
    print(f"Reserved {want} MiB on the GPU before training")
