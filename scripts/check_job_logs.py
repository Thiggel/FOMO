#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUMMARY_METRIC_RE = re.compile(
    r"\[.*?\]\[experiment\.utils\.print_mean_std\]\[INFO\]\s*-\s*(?P<metric>[^\n]+?)\s*-\s*"
    r"Mean:\s*(?P<mean>[-+0-9.eE]+),\s*Std:\s*(?P<std>[-+0-9.eE]+)",
    re.IGNORECASE,
)

TIMEOUT_RE = re.compile(
    r"DUE TO TIME LIMIT|CANCELLED AT|STEPD TERMINATED|JOB NOT ENDING",
    re.IGNORECASE,
)

FAILURE_PATTERNS = [
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bchildfailederror\b", re.IGNORECASE),
    re.compile(r"\bexperiment FAILED\b", re.IGNORECASE),
    re.compile(r"\bruntimeerror:\b", re.IGNORECASE),
    re.compile(r"\bvalueerror:\b", re.IGNORECASE),
    re.compile(r"\bdisk quota exceeded\b", re.IGNORECASE),
    re.compile(r"\bSIGKILL\b|\bexitcode\s*:\s*-9\b", re.IGNORECASE),
]

ARRAY_DIRECTIVE_RE = re.compile(r"^#SBATCH\s+--array=(.+)$", re.MULTILINE)
OVERRIDE_RE = re.compile(r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s\\]+)")
SLURM_SUFFIX_RE = re.compile(r"^(?P<job_id>\d+)(?:_(?P<task_id>\d+))?$")
SEED_DIR_RE = re.compile(r"seed_(\d+)$")


class LogAnalysis:
    def __init__(
        self,
        path: str,
        status: str,
        metric_count: int,
        metrics: Dict[str, Dict[str, float]],
        failures: List[str],
        last_metric_offset: int,
        last_failure_offset: int,
    ):
        self.path = path
        self.status = status
        self.metric_count = metric_count
        self.metrics = metrics
        self.failures = failures
        self.last_metric_offset = last_metric_offset
        self.last_failure_offset = last_failure_offset


def build_log_map(root: Path) -> Dict[str, Path]:
    job_logs_root = root / "job_logs"
    if not job_logs_root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p
        for p in job_logs_root.rglob("*")
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
        "stable_diffusion_3": "stable-diffusion-3",
        "vit_s": "vit-s",
        "vit_b": "vit-b",
        "mode_window": "mode-window",
        "ood_top": "ood-top",
    }
    for key, value in replacements.items():
        log_name = log_name.replace(key, value)
    return log_name


def candidate_logs(job_rel: str) -> List[str]:
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
    if job_rel == "jobs/environment.sh":
        candidates.append("job_logs/environment.out")

    unique = []  # type: List[str]
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def slurm_log_prefix(job_rel: str) -> str:
    rel_no_ext = job_rel.replace("jobs/", "").replace(".sh", "")
    return f"job_logs/{rel_no_ext}_"


def extract_slurm_ids(log_rel: str, prefix: str) -> Tuple[Optional[int], Optional[int]]:
    if not log_rel.startswith(prefix):
        return None, None
    suffix = log_rel[len(prefix) :]
    if suffix.endswith(".out"):
        suffix = suffix[: -len(".out")]
    match = SLURM_SUFFIX_RE.match(suffix)
    if not match:
        return None, None
    job_id = int(match.group("job_id"))
    task_id_raw = match.group("task_id")
    task_id = int(task_id_raw) if task_id_raw is not None else None
    return job_id, task_id


def resolve_job_logs(job_rel: str, log_files: Dict[str, Path]) -> Tuple[List[str], str]:
    exact_candidates = [c for c in candidate_logs(job_rel) if c in log_files]
    prefix = slurm_log_prefix(job_rel)
    slurm_matches = [
        rel for rel in log_files if rel.startswith(prefix) and rel.endswith(".out")
    ]

    if slurm_matches:
        grouped = {}  # type: Dict[int, List[str]]
        for rel in slurm_matches:
            job_id, _ = extract_slurm_ids(rel, prefix)
            if job_id is None:
                continue
            grouped.setdefault(job_id, []).append(rel)

        if grouped:
            latest_group_id = max(
                grouped,
                key=lambda group_id: max(
                    log_files[rel].stat().st_mtime for rel in grouped[group_id]
                ),
            )
            group_logs = grouped[latest_group_id]
            group_logs.sort(key=lambda rel: extract_slurm_ids(rel, prefix)[1] or -1)
            return group_logs, f"slurm_group:{latest_group_id}"

    if exact_candidates:
        exact_candidates.sort(key=lambda rel: log_files[rel].stat().st_mtime, reverse=True)
        return [exact_candidates[0]], "exact_latest"

    return [], "missing"


def read_tail_text(path: Path, tail_bytes: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        read_size = min(size, tail_bytes)
        if read_size > 0:
            handle.seek(-read_size, os.SEEK_END)
        return handle.read().decode("utf-8", errors="ignore")


def extract_summary_metrics(text: str) -> Tuple[Dict[str, Dict[str, float]], int]:
    metrics = {}  # type: Dict[str, Dict[str, float]]
    last_offset = -1
    for match in SUMMARY_METRIC_RE.finditer(text):
        metric_name = match.group("metric").strip()
        metrics[metric_name] = {
            "mean": float(match.group("mean")),
            "std": float(match.group("std")),
        }
        last_offset = match.end()
    return metrics, last_offset


def extract_failures(text: str) -> Tuple[List[str], int]:
    failures = []  # type: List[str]
    last_offset = -1

    for timeout_match in TIMEOUT_RE.finditer(text):
        failures.append("timeout_or_cancellation")
        last_offset = max(last_offset, timeout_match.end())

    for pattern in FAILURE_PATTERNS:
        for match in pattern.finditer(text):
            failures.append(pattern.pattern)
            last_offset = max(last_offset, match.end())

    deduped = sorted(set(failures))
    return deduped, last_offset


def analyze_log(path: Path, tail_bytes: int) -> LogAnalysis:
    tail_text = read_tail_text(path, tail_bytes=tail_bytes)
    metrics, last_metric_offset = extract_summary_metrics(tail_text)
    failures, last_failure_offset = extract_failures(tail_text)

    if metrics and last_metric_offset > last_failure_offset:
        status = "complete"
    elif failures:
        status = "failed"
    else:
        status = "incomplete"

    return LogAnalysis(
        path=path.as_posix(),
        status=status,
        metric_count=len(metrics),
        metrics=metrics,
        failures=failures,
        last_metric_offset=last_metric_offset,
        last_failure_offset=last_failure_offset,
    )


def parse_array_task_count(job_path: Path) -> Optional[int]:
    text = job_path.read_text(errors="ignore")
    match = ARRAY_DIRECTIVE_RE.search(text)
    if not match:
        return None

    spec = match.group(1).strip().split("%")[0]
    if not spec:
        return None

    count = 0
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            range_part, step_part = (token.split(":", 1) + ["1"])[:2]
            start_s, end_s = range_part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = int(step_part)
            if step <= 0:
                return None
            if end < start:
                return None
            count += ((end - start) // step) + 1
        else:
            int(token)
            count += 1
    return count


def parse_job_overrides(job_path: Path) -> dict:
    text = job_path.read_text(errors="ignore")
    overrides = {}  # type: Dict[str, str]
    for match in OVERRIDE_RE.finditer(text):
        key = match.group("key")
        value = match.group("value").strip().strip("'\"")
        overrides[key] = value

    parsed = {  # type: Dict[str, Any]
        "experiment_name": overrides.get("experiment_name"),
        "num_runs": None,
        "seeds": None,
    }

    if "num_runs" in overrides:
        try:
            parsed["num_runs"] = int(overrides["num_runs"])
        except ValueError:
            parsed["num_runs"] = None

    if "seeds" in overrides:
        raw = overrides["seeds"]
        match = re.match(r"\[(.*)\]$", raw)
        if match:
            seed_values = []
            for token in match.group(1).split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    seed_values.append(int(token))
                except ValueError:
                    continue
            if seed_values:
                parsed["seeds"] = seed_values

    return parsed


def load_default_run_config(root: Path) -> dict:
    config_path = root / "experiment" / "conf" / "config.yaml"
    defaults = {"num_runs": 3, "seeds": [0, 1, 2, 3, 4]}
    if not config_path.exists():
        return defaults

    text = config_path.read_text(errors="ignore")
    num_runs_match = re.search(r"^\s*num_runs:\s*(\d+)\s*$", text, re.MULTILINE)
    if num_runs_match:
        defaults["num_runs"] = int(num_runs_match.group(1))

    seeds_match = re.search(r"^\s*seeds:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
    if seeds_match:
        seeds = []
        for token in seeds_match.group(1).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                seeds.append(int(token))
            except ValueError:
                continue
        if seeds:
            defaults["seeds"] = seeds

    return defaults


def detect_base_cache_dir(root: Path) -> Optional[Path]:
    candidates = []

    env_base = os.environ.get("BASE_CACHE_DIR")
    if env_base:
        candidates.append(Path(env_base))

    work = os.environ.get("WORK")
    if work:
        candidates.append(Path(work) / "FOMO2")

    candidates.append(Path.home() / "FOMO2")
    candidates.append(root.parent / "FOMO2")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def aggregate_numeric_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not results:
        return {}

    common_keys = set(results[0].keys())
    for result in results[1:]:
        common_keys &= set(result.keys())

    aggregate = {}  # type: Dict[str, Dict[str, float]]
    for key in sorted(common_keys):
        values = []
        for result in results:
            value = result.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if len(values) != len(results):
            continue

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregate[key] = {"mean": mean, "std": math.sqrt(variance)}

    return aggregate


def load_result_json_aggregation(
    base_cache_dir: Optional[Path],
    overrides: dict,
    defaults: dict,
) -> Optional[dict]:
    if base_cache_dir is None:
        return None

    experiment_name = overrides.get("experiment_name")
    if not experiment_name:
        return None

    checkpoints_root = base_cache_dir / "checkpoints" / str(experiment_name)
    if not checkpoints_root.exists():
        return None

    num_runs = int(overrides.get("num_runs") or defaults["num_runs"])
    seed_order = overrides.get("seeds") or defaults["seeds"]
    expected_seeds = [int(seed) for seed in list(seed_order)[:num_runs]]

    dataset_summaries = []
    for dataset_dir in sorted(checkpoints_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        seed_results = {}  # type: Dict[int, Dict[str, Any]]
        newest_mtime = 0.0
        for seed_dir in sorted(dataset_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            seed_match = SEED_DIR_RE.match(seed_dir.name)
            if not seed_match:
                continue
            seed_value = int(seed_match.group(1))
            result_path = seed_dir / "result.json"
            if not result_path.exists():
                continue

            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    result_payload = json.load(handle)
            except Exception:
                continue

            seed_results[seed_value] = result_payload
            newest_mtime = max(newest_mtime, result_path.stat().st_mtime)

        if not seed_results:
            continue

        selected_seeds = [seed for seed in expected_seeds if seed in seed_results]
        selected_results = [seed_results[seed] for seed in selected_seeds]
        metrics = aggregate_numeric_results(selected_results)

        dataset_summaries.append(
            {
                "dataset_id": dataset_dir.name,
                "expected_seeds": expected_seeds,
                "available_seeds": sorted(seed_results.keys()),
                "selected_seeds": selected_seeds,
                "complete": set(expected_seeds).issubset(seed_results.keys()),
                "metrics": metrics,
                "metric_count": len(metrics),
                "_newest_mtime": newest_mtime,
            }
        )

    if not dataset_summaries:
        return None

    best_summary = max(
        dataset_summaries,
        key=lambda item: (
            item["complete"],
            len(item["selected_seeds"]),
            item["_newest_mtime"],
        ),
    )
    best_summary.pop("_newest_mtime", None)
    return best_summary


def classify_job_status(analyses: List[LogAnalysis], expected_task_count: Optional[int]) -> str:
    if not analyses:
        return "missing"

    completed = sum(1 for analysis in analyses if analysis.status == "complete")
    failed = sum(1 for analysis in analyses if analysis.status == "failed")

    if failed:
        return "failed"

    if expected_task_count is not None:
        if len(analyses) < expected_task_count:
            return "partial"
        if completed == expected_task_count:
            return "complete"
        if completed > 0:
            return "partial"
        return "incomplete"

    if completed == len(analyses):
        return "complete"
    if completed > 0:
        return "partial"
    return "incomplete"


def run_check(root: Path, json_out: Path, tail_bytes: int) -> int:
    defaults = load_default_run_config(root)
    base_cache_dir = detect_base_cache_dir(root)
    log_files = build_log_map(root)
    job_files = sorted(p for p in (root / "jobs").rglob("*.sh") if p.is_file())

    report_jobs = []
    completed_jobs = []  # type: List[Dict[str, Any]]
    not_completed_jobs = []  # type: List[Dict[str, Any]]
    aggregation_cache = {}  # type: Dict[Tuple[Optional[str], Optional[int], Tuple[int, ...]], Optional[Dict[str, Any]]]

    print("job\tlogs\tlog_status\tmetrics_in_logs\tagg_status\tagg_seeds")
    for job_path in job_files:
        job_rel = job_path.relative_to(root).as_posix()
        selected_log_rels, log_resolution = resolve_job_logs(job_rel, log_files)

        analyses = [
            analyze_log(log_files[log_rel], tail_bytes=tail_bytes)
            for log_rel in selected_log_rels
        ]
        expected_task_count = parse_array_task_count(job_path)
        expected_for_status = (
            expected_task_count if log_resolution.startswith("slurm_group:") else None
        )
        log_status = classify_job_status(analyses, expected_for_status)

        metrics_in_logs = sum(analysis.metric_count for analysis in analyses)

        overrides = parse_job_overrides(job_path)
        cache_key = (
            overrides.get("experiment_name"),
            overrides.get("num_runs"),
            tuple(overrides.get("seeds") or []),
        )
        if cache_key not in aggregation_cache:
            aggregation_cache[cache_key] = load_result_json_aggregation(
                base_cache_dir, overrides, defaults
            )
        aggregation = aggregation_cache[cache_key]
        if aggregation is None:
            agg_status = "N/A"
            agg_seeds = "-"
        else:
            agg_status = "COMPLETE" if aggregation["complete"] else "PARTIAL"
            agg_seeds = (
                f"{len(aggregation['selected_seeds'])}/{len(aggregation['expected_seeds'])}"
            )

        logs_cell = ",".join(selected_log_rels) if selected_log_rels else "MISSING"
        print(
            f"{job_rel}\t{logs_cell}\t{log_status}\t{metrics_in_logs}\t{agg_status}\t{agg_seeds}"
        )

        representative_metrics = {}
        for analysis in reversed(analyses):
            if analysis.status == "complete" and analysis.metrics:
                representative_metrics = analysis.metrics
                break

        report_jobs.append(
            {
                "job": job_rel,
                "log_resolution": log_resolution,
                "selected_logs": selected_log_rels,
                "expected_task_count": expected_task_count,
                "log_status": log_status,
                "metrics_in_logs": metrics_in_logs,
                "representative_log_metrics": representative_metrics,
                "logs": [
                    {
                        "path": analysis.path,
                        "status": analysis.status,
                        "metric_count": analysis.metric_count,
                        "failures": analysis.failures,
                        "last_metric_offset": analysis.last_metric_offset,
                        "last_failure_offset": analysis.last_failure_offset,
                    }
                    for analysis in analyses
                ],
                "aggregation": aggregation,
            }
        )
        job_record = report_jobs[-1]
        if log_status == "complete":
            completed_jobs.append(job_record)
        else:
            not_completed_jobs.append(job_record)

    completed_jobs.sort(key=lambda item: item["job"])
    not_completed_jobs.sort(key=lambda item: item["job"])

    print("\n=== Completed Jobs ===")
    if completed_jobs:
        for item in completed_jobs:
            print(item["job"])
    else:
        print("None")

    print("\n=== Not Completed Jobs ===")
    if not_completed_jobs:
        for item in not_completed_jobs:
            logs_cell = ",".join(item["selected_logs"]) if item["selected_logs"] else "MISSING"
            print(
                "{job}\tstatus={status}\tlogs={logs}".format(
                    job=item["job"],
                    status=item["log_status"],
                    logs=logs_cell,
                )
            )
    else:
        print("None")

    rerun_groups = {}  # type: Dict[str, List[str]]
    for item in not_completed_jobs:
        job_rel = item["job"]
        parent = str(Path(job_rel).parent)
        if parent == "jobs":
            continue
        rerun_groups.setdefault(parent, []).append(job_rel)

    for folder in rerun_groups:
        rerun_groups[folder] = sorted(set(rerun_groups[folder]))

    rerun_commands = []  # type: List[Dict[str, Any]]
    print("\n=== Commands To Re-run Not Completed Jobs By Folder ===")
    if rerun_groups:
        for folder in sorted(rerun_groups):
            jobs = rerun_groups[folder]
            command = "for job in {jobs}; do sbatch \"$job\"; done".format(
                jobs=" ".join(jobs)
            )
            print("{folder}/".format(folder=folder))
            print(command)
            rerun_commands.append(
                {
                    "folder": folder,
                    "jobs": jobs,
                    "command": command,
                }
            )
    else:
        print("No re-run commands needed.")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(),
        "base_cache_dir": base_cache_dir.as_posix() if base_cache_dir else None,
        "defaults": defaults,
        "jobs": report_jobs,
        "completed_jobs": [item["job"] for item in completed_jobs],
        "not_completed_jobs": [item["job"] for item in not_completed_jobs],
        "rerun_commands_by_folder": rerun_commands,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"\nJSON report written to: {json_out.as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strong job log checker with result extraction and JSON reporting."
    )
    parser.add_argument(
        "--json-out",
        default="job_logs/check_job_logs_report.json",
        help="Output path for the JSON report.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=256 * 1024,
        help="Number of trailing bytes to analyze per log file.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    return run_check(root=root, json_out=Path(args.json_out), tail_bytes=args.tail_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
