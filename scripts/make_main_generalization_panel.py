#!/usr/bin/env python3
"""Rebuild panel A of the in-paper generalization table from the full table.

The main-body table shows four of the seven downstream datasets so that it fits
the column width, and it was previously maintained by hand.  Two mutually
inconsistent SimCLR rows once survived into a submitted version that way.  Here
the numbers are read back out of the generated appendix table, so the two can
only disagree if this script is not run.

Panel B covers fine-tuning and segmentation, which come from different jobs, and
is left untouched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Column header in the full table, and the header to print here.
COLUMNS = [("Cars", "Cars"), ("Aircraft", "Aircraft"), ("Flowers", "Flowers"), ("IN100-LT", "IN100-LT")]

OBJECTIVES = ["SimCLR", "MoCo v3", "DINO", "DINOv2", "MAE"]


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
    return float(match.group(1)) if match else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    header, rows = parse(args.full_table)
    index = {name: header.index(name) for name, _ in COLUMNS}

    lines = [
        "\\multicolumn{6}{l}{\\textit{A. Frozen ViT-S transfer}} \\\\",
        "Objective & Method & " + " & ".join(out for _, out in COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for objective in OBJECTIVES:
        base = rows[f"{objective}, SSL baseline"]
        repair = rows[f"{objective}, with \\method"]
        rendered = []
        for name, _ in COLUMNS:
            i = index[name]
            better = value(repair[i]) > value(base[i])
            rendered.append(
                (
                    base[i] if better else f"\\mathbf{{{base[i].strip('$')}}}",
                    f"\\mathbf{{{repair[i].strip('$')}}}" if better else repair[i],
                )
            )
        lines.append(
            f"{objective} & SSL baseline & "
            + " & ".join(b if b.startswith("$") else f"${b}$" for b, _ in rendered)
            + " \\\\"
        )
        lines.append(
            " & With BRIDGE & "
            + " & ".join(r if r.startswith("$") else f"${r}$" for _, r in rendered)
            + " \\\\"
        )

    body = args.out.read_text()
    start = body.index("\\multicolumn{6}{l}{\\textit{A. Frozen ViT-S transfer}}")
    end = body.index("\\bottomrule", start)
    args.out.write_text(body[:start] + "\n".join(lines) + "\n" + body[end:])
    print(f"{args.out}: panel A rebuilt for {len(OBJECTIVES)} objectives")


if __name__ == "__main__":
    main()
