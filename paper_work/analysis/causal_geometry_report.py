"""Aggregate fixed-original-panel geometry snapshots from matched branches."""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse(path):
    # .../causal_geometry/<condition>/<seed>/generated/representation_diagnostics/cycle_N.json
    p = Path(path)
    generated = p.parents[1]
    return generated.parents[2].name, generated.parents[1].name, generated.parent.name


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--inputs", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    rows = []
    for raw in glob.glob(args.inputs, recursive=True):
        values = json.loads(Path(raw).read_text()); family, condition, seed = parse(raw)
        rows.append({"family":family, "condition":condition, "seed":seed, **values})
    if not rows:
        raise RuntimeError("No per-cycle representation diagnostics found")
    output=Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "causal_geometry_metrics.json").write_text(json.dumps(rows, indent=2))
    metric="representation/normalized_radius_median"
    plt.figure(figsize=(6,4))
    for condition in sorted(set(row["condition"] for row in rows)):
        for cycle in sorted(set(row["cycle"] for row in rows)):
            values=[row[metric] for row in rows if row["condition"]==condition and row["cycle"]==cycle]
            if values: plt.scatter([cycle], [np.mean(values)], label=condition if cycle==0 else None)
    plt.xlabel("training stage"); plt.ylabel("fixed-original normalized local radius"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(output/"fixed_panel_radius_by_branch.png",dpi=220)
    paired={}
    for row in rows: paired.setdefault((row["condition"],row["seed"]),{})[row["cycle"]]=row
    deltas=[]
    for (condition,seed), by_cycle in paired.items():
        if 0 in by_cycle and 1 in by_cycle:
            deltas.append({"condition":condition,"seed":seed,"delta_radius":by_cycle[1][metric]-by_cycle[0][metric],"delta_effective_rank":by_cycle[1]["representation/effective_rank"]-by_cycle[0]["representation/effective_rank"]})
    # Difference-in-differences is reported only for exact paired seeds.
    did=[]
    by_seed={}
    for row in deltas: by_seed.setdefault(row["seed"],{})[row["condition"]]=row
    for seed, entries in by_seed.items():
        for treatment in ("uniform_sd3","mode_sd3","top_sd3"):
            if treatment in entries and "no_repair" in entries:
                did.append({"seed":seed,"treatment":treatment,"did_radius_vs_no_repair":entries[treatment]["delta_radius"]-entries["no_repair"]["delta_radius"]})
    (output / "causal_geometry_difference_in_differences.json").write_text(json.dumps({"branch_deltas":deltas,"difference_in_differences":did},indent=2))


if __name__ == "__main__": main()
