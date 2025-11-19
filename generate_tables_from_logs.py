"""
Utility to parse job log files and build LaTeX-ready result tables.

Each job is expected to end with a short summary that lists mean and
standard deviation for each downstream task, e.g. lines such as:
    CIFAR10 - Mean: 0.76, Std: 0.03
    ImageNet - mean=0.42 std=0.02

The script searches a log directory recursively, extracts those values,
and writes both a raw CSV file and a LaTeX table to the output directory.
Missing values remain empty in the LaTeX table so it can still be
compiled even when some jobs did not finish.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - handled at runtime when dependencies are missing
    raise SystemExit(
        "pandas is required for table generation. Install it with `pip install pandas`."
    ) from exc


SUMMARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?P<task>[\w\-/ .:+]+?)\s*[-:]\s*Mean[:=]\s*(?P<mean>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
        r"(?:[,;\s]+Std[:=]\s*(?P<std>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?P<task>[\w\-/ .:+]+?)\s+mean[:=]\s*(?P<mean>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
        r"[,;\s]+std[:=]\s*(?P<std>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        flags=re.IGNORECASE,
    ),
]


def parse_log_file(path: Path) -> Dict[str, Dict[str, float]]:
    """Return mapping of task -> {"mean": float, "std": float} for a log file."""
    text = path.read_text(errors="ignore")
    results: Dict[str, Dict[str, float]] = {}

    for line in text.splitlines():
        for pattern in SUMMARY_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue

            task = match.group("task").strip()
            mean = float(match.group("mean"))
            std = float(match.group("std")) if match.group("std") is not None else None
            if std is None:
                continue

            results[task] = {"mean": mean, "std": std}
            break  # avoid matching same line multiple times

    return results


def collect_results(log_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk through *log_dir* and aggregate parsed summaries keyed by log stem."""
    collected: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not log_dir.exists():
        return collected

    for log_file in log_dir.rglob("*"):
        if not log_file.is_file():
            continue

        parsed = parse_log_file(log_file)
        if parsed:
            job_name = log_file.relative_to(log_dir).with_suffix("").as_posix()
            collected[job_name] = parsed

    return collected


def build_tables(results: Mapping[str, Mapping[str, Mapping[str, float]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build tables from parsed results.

    Returns a tuple of `(wide_numeric, formatted_strings)` where:
    - `wide_numeric` uses a MultiIndex column of (job, metric) with mean/std floats.
    - `formatted_strings` stores `mean \\pm std` strings for LaTeX output.
    """
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    tasks = sorted({task for job in results.values() for task in job})
    jobs = sorted(results)

    multi_columns = pd.MultiIndex.from_product([jobs, ["mean", "std"]])
    numeric_rows = []
    formatted_rows = []

    for task in tasks:
        numeric_row = []
        formatted_row = {}
        for job in jobs:
            metrics = results[job].get(task)
            if metrics:
                numeric_row.extend([metrics.get("mean"), metrics.get("std")])
                formatted_row[job] = f"{metrics['mean']:.3f} \\pm {metrics['std']:.3f}"
            else:
                numeric_row.extend([pd.NA, pd.NA])
                formatted_row[job] = ""
        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    wide_numeric = pd.DataFrame(numeric_rows, index=tasks, columns=multi_columns)
    formatted_strings = pd.DataFrame(formatted_rows, index=tasks)

    wide_numeric.index.name = "task"
    formatted_strings.index.name = "task"

    return wide_numeric, formatted_strings


def write_outputs(
    wide_numeric: pd.DataFrame,
    formatted_strings: pd.DataFrame,
    output_dir: Path,
    table_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / f"{table_name}_raw.csv"
    latex_table = output_dir / f"{table_name}.tex"

    wide_numeric.to_csv(raw_csv)
    formatted_strings.to_latex(latex_table, escape=False, na_rep="")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("job_logs"),
        help="Directory containing job log files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Tables"),
        help="Directory where generated tables are stored.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="results",
        help="Base name for the generated table files (without extension).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = collect_results(args.log_dir)

    wide_numeric, formatted_strings = build_tables(results)
    write_outputs(wide_numeric, formatted_strings, args.output_dir, args.table_name)

    print(f"Parsed {len(results)} job logs.")
    print(f"Tasks found: {', '.join(formatted_strings.index)}")
    print(f"Tables written to {args.output_dir.resolve()} with base name '{args.table_name}'.")


if __name__ == "__main__":
    main()
