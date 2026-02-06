#!/usr/bin/env python3
import os
import re
from pathlib import Path


def build_log_map(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p
        for p in (root / "job_logs").rglob("*")
        if p.is_file()
    }


def ablation_log_name(category: str, name: str) -> str:
    if category == "cycles":
        log_name = name
    else:
        log_name = f"{category}_{name}"

    replacements = {
        "sample_selection": "sample-selection",
        "stable_diffusion": "stable-diffusion",
        "stable-diffusion_2": "stable-diffusion-2",
        "stable-diffusion_3": "stable-diffusion-3",
        "vit_s": "vit-s",
        "vit_b": "vit-b",
        "mode_window": "mode-window",
        "ood_top": "ood-top",
    }
    for key, value in replacements.items():
        log_name = log_name.replace(key, value)
    return log_name


def candidate_logs(job_rel: str) -> list:
    parts = job_rel.split("/")
    name = Path(job_rel).stem
    candidates = [job_rel.replace("jobs/", "job_logs/").replace(".sh", ".out")]

    if job_rel.startswith("jobs/sota/") and len(parts) >= 4:
        dataset = parts[2]
        candidates.append(f"job_logs/{dataset}/{name}.out")
    if job_rel.startswith("jobs/ablations/") and len(parts) >= 4:
        category = parts[2]
        log_name = ablation_log_name(category, name)
        candidates.append(f"job_logs/ablations/{log_name}.out")
    if job_rel.startswith("jobs/baseline/"):
        candidates.append(f"job_logs/baseline/{name}.out")
    if job_rel == "jobs/install_environment.sh":
        candidates.append("job_logs/install_environment.out")

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def tail_has_mean_std(path: Path) -> bool:
    re_mean = re.compile(r"\bmean\b", re.IGNORECASE)
    re_std = re.compile(r"\bstd\b", re.IGNORECASE)
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, 16384)
            f.seek(-read_size, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="ignore")
        return bool(re_mean.search(tail) and re_std.search(tail))
    except Exception:
        return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    job_files = sorted(p for p in (root / "jobs").rglob("*.sh") if p.is_file())
    log_files = build_log_map(root)

    print("job\tlog\tmean_std_at_end")
    for job in job_files:
        rel = job.relative_to(root).as_posix()
        candidates = candidate_logs(rel)
        existing = [c for c in candidates if c in log_files]
        log_path = existing[0] if existing else None

        if log_path:
            has_mean_std = tail_has_mean_std(log_files[log_path])
            mean_std = str(has_mean_std)
        else:
            mean_std = "N/A"

        print(f"{rel}\t{log_path or 'MISSING'}\t{mean_std}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
