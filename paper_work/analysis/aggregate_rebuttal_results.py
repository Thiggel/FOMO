"""Create a traceable paper-ready ledger from per-seed result/protocol files."""
import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--output", required=True); args=parser.parse_args()
    rows=[]
    for result_path in glob.glob(str(Path(args.root) / "**" / "result.json"), recursive=True):
        result_file=Path(result_path); protocol_file=result_file.with_name("protocol.json")
        try: result=json.loads(result_file.read_text())
        except json.JSONDecodeError: continue
        protocol=json.loads(protocol_file.read_text()) if protocol_file.exists() else {}
        rows.append({"checkpoint_dir":str(result_file.parent),"protocol":protocol,"results":result})
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/"all_seed_results.json").write_text(json.dumps(rows,indent=2))
    lines=["# Rebuttal experiment ledger", "", "| Experiment | Seed | Updates | Hours | Key metrics |", "|---|---:|---:|---:|---|"]
    for row in sorted(rows,key=lambda item:item["checkpoint_dir"]):
        protocol,row_result=row["protocol"],row["results"]
        schedule=protocol.get("schedule",{})
        key_metrics=[f"{key}={value:.3f}" for key,value in row_result.items() if isinstance(value,float) and ("acc" in key.lower() or "loss" in key.lower())][:4]
        lines.append("| {} | {} | {} | {:.2f} | {} |".format(
            Path(row["checkpoint_dir"]).parts[-3] if len(Path(row["checkpoint_dir"]).parts)>=3 else row["checkpoint_dir"],
            protocol.get("seed","?"), schedule.get("nominal_total_optimizer_updates","?"), float(row_result.get("training_time",0.0)), "; ".join(key_metrics)))
    (output/"experiment_ledger.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__": main()
