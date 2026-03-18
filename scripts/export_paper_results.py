#!/usr/bin/env python3
"""Aggregate paper-ready metrics from the final successful job logs.

This script is designed for the current FOMO paper workflow:
- it scans the experiment job scripts we care about (excluding ImageNet-1k),
- finds all matching Slurm array logs,
- picks the latest *successful* log per seed/task,
- parses the final `print_mean_std` summaries from each selected log,
- aggregates mean/std across seeds, and
- writes JSON/CSV artifacts for downstream LaTeX table generation.

The per-seed selection is important because some experiments were rerun only for
individual failed seeds. In those cases the final 3-seed result can span
multiple Slurm submissions.
"""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


SUMMARY_METRIC_RE = re.compile(
    r"\[.*?\]\[experiment\.utils\.print_mean_std\]\[INFO\]\s*-\s*(?P<metric>[^\n]+?)\s*-\s*"
    r"Mean:\s*(?P<mean>[-+0-9.eE]+),\s*Std:\s*(?P<std>[-+0-9.eE]+)",
    re.IGNORECASE,
)

FAILURE_RE = re.compile(
    r"traceback \(most recent call last\)|\bchildfailederror\b|\bexperiment FAILED\b|"
    r"\bruntimeerror:\b|\bvalueerror:\b|\bSIGKILL\b|\bWatchdog caught collective operation timeout\b|"
    r"DUE TO TIME LIMIT|CANCELLED AT",
    re.IGNORECASE,
)

ARRAY_DIRECTIVE_RE = re.compile(r"^#SBATCH\s+--array=(.+)$", re.MULTILINE)
SLURM_LOG_RE = re.compile(r"^(?P<job_id>\d+)(?:_(?P<task_id>\d+))?$")


TARGET_GLOBS = [
    "jobs/baseline/*.sh",
    "jobs/ablations/*/*.sh",
    "jobs/sota/imagenet-100-lt/*.sh",
    "jobs/sota/cifar-10-lt/*.sh",
    "jobs/sota/cifar-100-lt/*.sh",
    "jobs/sota/pass-subset/*.sh",
    "jobs/sota/diffusiondb-subset/*.sh",
]


DEFAULT_BRIDGE_SOURCE_JOB = "jobs/ablations/sample_selection/mode_window_q75_diverse.sh"
DEFAULT_BRIDGE_ALIAS_JOBS = [
    "jobs/baseline/newmethod_imbalanced.sh",
    "jobs/ablations/pretraining/simclr.sh",
    "jobs/ablations/architecture/resnet50.sh",
    "jobs/ablations/sample_selection/mode_window.sh",
]


class ParsedLog(object):
    def __init__(self, path, job_id, task_id, metrics, complete):
        self.path = path
        self.job_id = job_id
        self.task_id = task_id
        self.metrics = metrics
        self.complete = complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_work/results_export"),
        help="Directory for JSON/CSV artifacts.",
    )
    return parser.parse_args()


def iter_target_jobs(root: Path) -> List[Path]:
    jobs: List[Path] = []
    for pattern in TARGET_GLOBS:
        jobs.extend(sorted(root.glob(pattern)))
    return jobs


def build_log_index(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for log_path in (root / "job_logs").rglob("*.out"):
        rel = log_path.relative_to(root).as_posix()
        index.setdefault(rel, []).append(log_path)
    return index


def chunked(items: List[int], size: int) -> List[List[int]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def parse_expected_task_count(job_path: Path) -> int:
    text = job_path.read_text(errors="ignore")
    match = ARRAY_DIRECTIVE_RE.search(text)
    if not match:
        return 1

    spec = match.group(1).strip().split("%")[0]
    if not spec:
        return 1

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
            count += ((end - start) // step) + 1
        else:
            count += 1
    return max(count, 1)


def slurm_log_prefix(job_rel: str) -> str:
    return "job_logs/" + job_rel.replace("jobs/", "").replace(".sh", "") + "_"


def ablation_log_name(category: str, name: str) -> str:
    log_name = name if category == "cycles" else "{0}_{1}".format(category, name)
    replacements = {
        "sample_selection": "sample-selection",
        "stable_diffusion": "stable-diffusion",
        "stable_diffusion_2": "stable-diffusion-2",
        "stable_diffusion_3": "stable-diffusion-3",
        "vit_s": "vit-s",
        "vit_b": "vit-b",
        "mode_window": "mode-window",
        "ood_top": "ood-top",
    }
    for key, value in replacements.items():
        log_name = log_name.replace(key, value)
    return log_name


def exact_candidate_logs(job_rel: str) -> List[str]:
    parts = job_rel.split("/")
    name = Path(job_rel).stem
    candidates = ["job_logs/" + job_rel.replace("jobs/", "").replace(".sh", ".out")]

    if job_rel.startswith("jobs/sota/") and len(parts) >= 4:
        dataset = parts[2]
        candidates.append("job_logs/{0}/{1}.out".format(dataset, name))
    if job_rel.startswith("jobs/ablations/") and len(parts) >= 4:
        category = parts[2]
        candidates.append(
            "job_logs/ablations/{0}.out".format(ablation_log_name(category, name))
        )
    if job_rel.startswith("jobs/baseline/"):
        candidates.append("job_logs/baseline/{0}.out".format(name))

    unique: List[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def parse_log_ids(log_name: str, prefix: str) -> Tuple[Optional[int], int]:
    suffix = log_name[len(prefix) :]
    if suffix.endswith(".out"):
        suffix = suffix[:-4]
    match = SLURM_LOG_RE.match(suffix)
    if not match:
        return None, 0
    job_id_raw = match.group("job_id")
    task_id_raw = match.group("task_id")
    return int(job_id_raw), int(task_id_raw or 0)


def fetch_slurm_states(job_ids: List[int]) -> Dict[Tuple[int, int], str]:
    states: Dict[Tuple[int, int], str] = {}
    if not job_ids:
        return states

    for group in chunked(sorted(set(job_ids)), 200):
        cmd = [
            "sacct",
            "-j",
            ",".join(str(job_id) for job_id in group),
            "--format=JobID,State",
            "-P",
            "-n",
        ]
        output = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
        for line in output.splitlines():
            if not line.strip():
                continue
            job_id_raw, state = (line.split("|", 1) + [""])[:2]
            if "." in job_id_raw:
                continue
            match = SLURM_LOG_RE.match(job_id_raw)
            if not match:
                continue
            job_id = int(match.group("job_id"))
            task_id = int(match.group("task_id") or 0)
            states[(job_id, task_id)] = state.strip()
    return states


def read_tail_text(path: Path, tail_bytes: int):
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        read_size = min(size, tail_bytes)
        if read_size > 0:
            handle.seek(-read_size, os.SEEK_END)
        return handle.read().decode("utf-8", errors="ignore")


def parse_metrics_from_tail(path: Path) -> ParsedLog:
    text = read_tail_text(path, tail_bytes=256 * 1024)
    metrics: Dict[str, Dict[str, float]] = {}
    last_metric_end = -1
    for match in SUMMARY_METRIC_RE.finditer(text):
        metric = match.group("metric").strip()
        metrics[metric] = {
            "mean": float(match.group("mean")),
            "std": float(match.group("std")),
        }
        last_metric_end = match.end()

    failure_match = None
    for failure_match in FAILURE_RE.finditer(text):
        pass

    complete = bool(metrics) and (
        failure_match is None or last_metric_end > failure_match.end()
    )
    return ParsedLog(path=path, job_id=None, task_id=0, metrics=metrics, complete=complete)


def filter_metric_name(metric: str) -> bool:
    if "_class_" in metric:
        return False
    if metric.endswith("_loss"):
        return False
    if metric == "training_time":
        return True
    if metric.endswith("_accuracy"):
        return True
    return False


def collect_job_metrics(
    root: Path,
    job_path: Path,
    log_index: Dict[str, List[Path]],
    slurm_states: Dict[Tuple[int, int], str],
) -> dict:
    job_rel = job_path.relative_to(root).as_posix()
    prefix = slurm_log_prefix(job_rel)
    expected_task_count = parse_expected_task_count(job_path)

    matching_logs: List[Path] = []
    for rel, paths in log_index.items():
        if rel.startswith(prefix):
            matching_logs.extend(paths)
    matching_logs.sort()

    selected: Dict[int, ParsedLog] = {}
    for log_path in matching_logs:
        rel = log_path.relative_to(root).as_posix()
        job_id, task_id = parse_log_ids(rel, prefix)
        if job_id is None:
            continue
        if slurm_states.get((job_id, task_id)) != "COMPLETED":
            continue

        parsed = parse_metrics_from_tail(log_path)
        parsed.job_id = job_id
        parsed.task_id = task_id
        if not parsed.metrics:
            continue

        previous = selected.get(task_id)
        if previous is None:
            selected[task_id] = parsed
            continue

        prev_key = (
            previous.job_id or -1,
            previous.path.stat().st_mtime,
        )
        curr_key = (
            parsed.job_id or -1,
            parsed.path.stat().st_mtime,
        )
        if curr_key > prev_key:
            selected[task_id] = parsed

    task_ids = sorted(selected)
    metrics_by_task = {task_id: selected[task_id].metrics for task_id in task_ids}

    aggregate: Dict[str, dict] = {}
    metric_names = sorted(
        {
            metric
            for task_metrics in metrics_by_task.values()
            for metric in task_metrics
            if filter_metric_name(metric)
        }
    )
    for metric in metric_names:
        values = [
            task_metrics[metric]["mean"]
            for task_metrics in metrics_by_task.values()
            if metric in task_metrics
        ]
        if not values:
            continue
        aggregate[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "num_seeds": len(values),
        }

    return {
        "job": job_rel,
        "expected_task_count": expected_task_count,
        "selected_task_ids": task_ids,
        "selected_logs": {
            str(task_id): selected[task_id].path.relative_to(root).as_posix()
            for task_id in task_ids
        },
        "complete": len(task_ids) == expected_task_count,
        "aggregate_metrics": aggregate,
    }


def collect_exact_job_metrics(root: Path, job_path: Path) -> dict:
    job_rel = job_path.relative_to(root).as_posix()
    expected_task_count = parse_expected_task_count(job_path)

    best_path: Optional[Path] = None
    best_log: Optional[ParsedLog] = None
    for candidate in exact_candidate_logs(job_rel):
        candidate_path = root / candidate
        if not candidate_path.exists():
            continue
        parsed = parse_metrics_from_tail(candidate_path)
        if not parsed.metrics:
            continue
        if best_path is None or candidate_path.stat().st_mtime > best_path.stat().st_mtime:
            best_path = candidate_path
            best_log = parsed

    if best_path is None or best_log is None:
        return {
            "job": job_rel,
            "expected_task_count": expected_task_count,
            "selected_task_ids": [],
            "selected_logs": {},
            "complete": False,
            "aggregate_metrics": {},
        }

    aggregate: Dict[str, dict] = {}
    for metric, stats in best_log.metrics.items():
        if not filter_metric_name(metric):
            continue
        aggregate[metric] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "num_seeds": expected_task_count,
        }

    return {
        "job": job_rel,
        "expected_task_count": expected_task_count,
        "selected_task_ids": list(range(expected_task_count)),
        "selected_logs": {"exact": best_path.relative_to(root).as_posix()},
        "complete": True,
        "aggregate_metrics": aggregate,
    }


def metric_display_name(metric: str) -> str:
    mapping = {
        "cifar10r_test_accuracy": "CIFAR-10-LT Lin.",
        "cifar10knn_knn_test_accuracy": "CIFAR-10-LT kNN",
        "cifar100r_test_accuracy": "CIFAR-100-LT Lin.",
        "cifar100knn_knn_test_accuracy": "CIFAR-100-LT kNN",
        "cars_test_accuracy": "Cars Lin.",
        "carsknn_knn_test_accuracy": "Cars kNN",
        "aircraft_test_accuracy": "Aircraft Lin.",
        "aircraftknn_knn_test_accuracy": "Aircraft kNN",
        "flowers_test_accuracy": "Flowers Lin.",
        "flowersknn_knn_test_accuracy": "Flowers kNN",
        "pets_test_accuracy": "Pets Lin.",
        "petsknn_knn_test_accuracy": "Pets kNN",
        "imagenet100lt_test_accuracy": "ImageNet-100-LT Lin.",
        "imagenet100ltknn_knn_test_accuracy": "ImageNet-100-LT kNN",
        "training_time": "Training time (h)",
    }
    return mapping.get(metric, metric)


def experiment_label(job_rel: str) -> str:
    parts = Path(job_rel).parts
    stem = Path(job_rel).stem
    if "baseline" in parts:
        return f"baseline/{stem}"
    if "ablations" in parts:
        return f"ablations/{parts[2]}/{stem}"
    if "sota" in parts:
        return f"sota/{parts[2]}/{stem}"
    return job_rel


def write_outputs(results: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "paper_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    csv_path = output_dir / "paper_results_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "job",
                "metric",
                "metric_display",
                "mean",
                "std",
                "num_seeds",
                "complete",
            ]
        )
        for result in results:
            for metric, stats in sorted(result["aggregate_metrics"].items()):
                writer.writerow(
                    [
                        experiment_label(result["job"]),
                        result["job"],
                        metric,
                        metric_display_name(metric),
                        f"{stats['mean']:.10f}",
                        f"{stats['std']:.10f}",
                        stats["num_seeds"],
                        result["complete"],
                    ]
                )


def alias_default_bridge_results(results: List[dict]) -> List[dict]:
    """Propagate the promoted default BRIDGE selector into protocol-matched rows.

    The completed Q75-diverse run is the current default BRIDGE configuration on
    ImageNet-100-LT. Several paper rows correspond to that same protocol under
    different ablation/baseline labels, so we alias their exported metrics to
    the new source result instead of editing downstream LaTeX tables by hand.
    """

    source_result: Optional[dict] = None
    for result in results:
        if result["job"] == DEFAULT_BRIDGE_SOURCE_JOB and result["complete"]:
            source_result = result
            break

    if source_result is None:
        return results

    aliased: List[dict] = []
    for result in results:
        if result["job"] not in DEFAULT_BRIDGE_ALIAS_JOBS:
            aliased.append(result)
            continue

        replacement = dict(result)
        replacement["selected_task_ids"] = list(source_result["selected_task_ids"])
        replacement["selected_logs"] = dict(source_result["selected_logs"])
        replacement["complete"] = source_result["complete"]
        replacement["aggregate_metrics"] = {
            metric: dict(stats)
            for metric, stats in source_result["aggregate_metrics"].items()
        }
        replacement["aliased_from"] = DEFAULT_BRIDGE_SOURCE_JOB
        aliased.append(replacement)
    return aliased


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    log_index = build_log_index(root)
    relevant_job_ids: List[int] = []
    for job_path in iter_target_jobs(root):
        prefix = slurm_log_prefix(job_path.relative_to(root).as_posix())
        for rel in log_index:
            if not rel.startswith(prefix):
                continue
            job_id, _ = parse_log_ids(rel, prefix)
            if job_id is not None:
                relevant_job_ids.append(job_id)
    slurm_states = fetch_slurm_states(relevant_job_ids)
    results = [
        collect_job_metrics(root, job_path, log_index, slurm_states)
        for job_path in iter_target_jobs(root)
    ]
    resolved_results = []
    for result in results:
        if result["complete"] or result["aggregate_metrics"]:
            resolved_results.append(result)
        else:
            resolved_results.append(
                collect_exact_job_metrics(root, root / result["job"])
            )
    resolved_results = alias_default_bridge_results(resolved_results)
    write_outputs(resolved_results, args.output_dir)

    complete = sum(1 for result in resolved_results if result["complete"])
    print(f"Wrote {len(resolved_results)} experiment summaries to {args.output_dir}")
    print(f"Complete experiments: {complete}/{len(resolved_results)}")


if __name__ == "__main__":
    main()
