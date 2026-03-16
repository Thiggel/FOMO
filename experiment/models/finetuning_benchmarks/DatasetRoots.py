import os


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def allow_data_downloads() -> bool:
    """Allow benchmark datasets to download missing files when explicitly enabled."""
    return _as_bool(os.getenv("FOMO_ALLOW_DATA_DOWNLOAD"), default=False)


def _resolve(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def get_data_root() -> str:
    explicit_root = os.getenv("FOMO_DATA_ROOT")
    if explicit_root:
        data_root = _resolve(explicit_root)
    else:
        base_cache_dir = os.getenv("BASE_CACHE_DIR")
        if not base_cache_dir:
            raise EnvironmentError(
                "Neither FOMO_DATA_ROOT nor BASE_CACHE_DIR is set; cannot locate benchmark data root."
            )
        data_root = _resolve(os.path.join(base_cache_dir, "data"))

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Benchmark data root does not exist: {data_root}. "
            f"FOMO_DATA_ROOT={os.getenv('FOMO_DATA_ROOT')!r}, BASE_CACHE_DIR={os.getenv('BASE_CACHE_DIR')!r}"
        )
    return data_root


def get_stanford_cars_root() -> str:
    explicit_root = os.getenv("STANFORD_CARS_ROOT")
    if explicit_root:
        cars_root = _resolve(explicit_root)
    else:
        base_cache_dir = os.getenv("BASE_CACHE_DIR")
        if not base_cache_dir:
            raise EnvironmentError(
                "Neither STANFORD_CARS_ROOT nor BASE_CACHE_DIR is set; cannot locate Stanford Cars dataset."
            )
        cars_root = _resolve(os.path.join(base_cache_dir, "stanford_cars"))

    if not os.path.isdir(cars_root):
        raise FileNotFoundError(
            f"Stanford Cars root does not exist: {cars_root}. "
            f"STANFORD_CARS_ROOT={os.getenv('STANFORD_CARS_ROOT')!r}, BASE_CACHE_DIR={os.getenv('BASE_CACHE_DIR')!r}"
        )
    return cars_root
