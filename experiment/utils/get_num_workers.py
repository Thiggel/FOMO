import os


def get_num_workers():
    override = os.getenv("FOMO_NUM_WORKERS")
    if override is not None:
        try:
            override_value = int(override)
        except ValueError:
            override_value = 0
        return max(0, override_value)

    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count

    if max_num_worker_suggest is None:
        return 0

    num_workers = min(12, int(max_num_worker_suggest * 0.75))

    return num_workers
