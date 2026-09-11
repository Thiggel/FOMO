#!/usr/bin/env python3
"""Rebuild the in-paper policy control table from the generated full table.

Companion to make_main_generalization_panel.py, and it exists for the same
reason: the main-body table shows four of the seven downstream datasets so that
it fits the page, and maintaining that by hand is how a stale row survives a
revision.  The numbers here are read back out of the appendix table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

COLUMNS = ["Cars", "Aircraft", "Flowers", "IN100-LT"]

# Row label in the full table, and the label to print here.  Order is the order
# of the printed table: the loop and its two closest controls first, then the
# acquisition alternatives, then the generation alternatives.
ROWS = [
    ("No repair", "No repair"),
    ("Adaptive \\method", "Adaptive BRIDGE"),
    ("Frozen first-cycle selector", "Frozen first-cycle selector"),
    ("One-shot repair", "One-shot repair"),
    ("Uniform acquisition", "Uniform acquisition"),
    ("Top-tail acquisition", "Top-tail acquisition"),
    ("Mode-window conventional augmentation", "Mode-window conventional augmentation"),
    (
        "AIDE-style VLM acquisition and text-to-image",
        "AIDE-style VLM acquisition and text-to-image",
    ),
    ("Mode-window caption and text-to-image", "Mode-window caption and text-to-image"),
    ("Mode-window captioned image-to-image", "Mode-window captioned image-to-image"),
]


def parse(table: Path) -> tuple[list[str], dict[str, list[str]]]:
    header: list[str] = []
    rows: dict[str, list[str]] = {}
    for line in table.read_text().splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        if cells[0] == "Method":
            header = cells[1:]
            continue
        if cells[0].startswith("\\") or len(cells) < 2:
            continue
        rows[cells[0]] = cells[1:]
    return header, rows


def value(cell: str) -> float:
    match = re.search(r"([0-9.]+)\\pm", cell)
    return float(match.group(1)) if match else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    header, rows = parse(args.full_table)
    index = {name: header.index(name) for name in COLUMNS}

    best = {
        name: max(value(rows[src][index[name]]) for src, _ in ROWS if src in rows)
        for name in COLUMNS
    }

    lines = [f"Method & {' & '.join(COLUMNS)} \\\\", "\\midrule"]
    for source, label in ROWS:
        cells = []
        for name in COLUMNS:
            cell = rows[source][index[name]]
            if value(cell) == best[name]:
                cell = f"$\\mathbf{{{cell.strip('$')}}}$"
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    body = args.out.read_text()
    start = body.index("Method & ")
    end = body.index("\\bottomrule", start)
    args.out.write_text(body[:start] + "\n".join(lines) + "\n" + body[end:])
    print(f"{args.out}: rebuilt with {len(ROWS)} rows")


if __name__ == "__main__":
    main()
