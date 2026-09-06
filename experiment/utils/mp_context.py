"""Which multiprocessing start method dataloader workers use.

``spawn`` re-creates each worker from scratch, and a worker rebuilds the
parent's synchronisation primitives by name: ``SemLock._rebuild`` calls
``sem_open`` on a POSIX semaphore in /dev/shm.  On the gruenau nodes
systemd-logind runs with RemoveIPC=yes, so it deletes every IPC object owned by
this uid the moment any login session of ours on the node ends.  A worker pool
respawned after that sweep dies with ``FileNotFoundError`` out of
``SemLock._rebuild``, hours into a run and through no fault of the run.  Three
jobs on gruenau7 went within fifteen seconds of each other this way.

``fork`` children inherit the semaphores directly and never look them up by
name, so an unlink cannot reach them.  It is also torch's own default on Linux.
The workers here decode and augment images and never touch CUDA, which is the
condition that makes forking after CUDA initialisation safe.
"""

import os


def start_method() -> str:
    return os.environ.get("FOMO_START_METHOD", "spawn")
