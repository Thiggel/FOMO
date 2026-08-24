#!/usr/bin/env python3
"""Report experiment conditions that produced bit-identical encoders.

An ablation row is only evidence if the run it summarises actually differs
from its neighbours.  Two failure modes produce rows that look like results
but are not:

* a knob that is a mathematically monotonic no-op, so the selector picks the
  same samples under both settings, and
* a run that stopped before the knob could take effect, leaving an encoder
  from a cycle the two conditions share.

Both show up the same way -- identical checkpoint weights under different
condition names -- and both are invisible in the tables, because the linear
probe is stochastic and prints different numbers for the same encoder.  The
kNN columns are the tell: kNN is deterministic given an encoder, so identical
kNN across two rows means one encoder.

Run this before regenerating tables.  Exit status is non-zero when a
collision is found so it can gate a release.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import torch


def encoder_fingerprint(checkpoint: Path) -> str | None:
    """Hash every floating-point tensor in a checkpoint's state dict."""
    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    state = blob.get("state_dict", blob)
    if not hasattr(state, "items"):
        return None
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value) and value.is_floating_point():
            digest.update(key.encode())
            digest.update(value.float().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument(
        "--expected-duplicate",
        action="append",
        default=[],
        metavar="COND",
        help=(
            "Condition name that is a deliberate alias of the default "
            "configuration.  Repeat per condition.  Collisions consisting "
            "only of these are reported as expected rather than as faults."
        ),
    )
    args = parser.parse_args()

    by_fingerprint: dict[tuple[str, str], set[str]] = defaultdict(set)
    scanned = 0
    for checkpoint in sorted(args.checkpoint_root.glob("*/*/seed_*/last.ckpt")):
        condition = checkpoint.relative_to(args.checkpoint_root).parts[0]
        seed = checkpoint.parent.name
        fingerprint = encoder_fingerprint(checkpoint)
        if fingerprint is None:
            print(f"  unreadable: {checkpoint}", file=sys.stderr)
            continue
        scanned += 1
        by_fingerprint[(seed, fingerprint)].add(condition)

    expected = set(args.expected_duplicate)
    faults = []
    benign = []
    for (seed, _), conditions in sorted(by_fingerprint.items()):
        if len(conditions) < 2:
            continue
        (benign if conditions <= expected else faults).append((seed, sorted(conditions)))

    print(f"scanned {scanned} checkpoints")
    if benign:
        print(f"\n{len(benign)} expected alias collisions:")
        for seed, conditions in benign:
            print(f"  {seed}: {', '.join(conditions)}")
    if faults:
        print(f"\n{len(faults)} UNEXPECTED identical-encoder collisions:")
        for seed, conditions in faults:
            print(f"  {seed}: {', '.join(conditions)}")
        print(
            "\nEach group above is one encoder reported under several condition "
            "names.  Those rows are not independent evidence."
        )
        return 1
    print("\nno unexpected collisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
