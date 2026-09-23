"""Dump per-seed DeepLog/TCN F1 and summarize identical-decision note."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    dl = load("logs/deeplog_results.json")
    print("=== HDFS-ish DeepLog vs TCN (from logs/deeplog_results.json; max_rows=200k) ===")
    by = {}
    for r in dl["rows"]:
        by.setdefault(r["Seed"], {})[r["Method"]] = r["F1"]
    for seed in sorted(by):
        d = by[seed]
        print(
            f"seed={seed}  DL-topk={d.get('DeepLog (top-k binary)', float('nan')):.4f}  "
            f"TCN-topk={d.get('TCN (top-k binary)', float('nan')):.4f}  "
            f"DL-nll={d.get('DeepLog (NLL threshold)', float('nan')):.4f}  "
            f"TCN-nll={d.get('TCN (NLL threshold)', float('nan')):.4f}"
        )

    hu = load("logs/huflit_results.json")
    print("\n=== HUFLIT DeepLog/TCN ===")
    by = {}
    for r in hu["rows"]:
        by.setdefault(r["Seed"], {})[r["Method"]] = r["F1"]
    for seed in sorted(by):
        d = by[seed]
        keys = sorted(d)
        print(f"seed={seed}")
        for k in keys:
            if "DeepLog" in k or "TCN" in k:
                print(f"  {k}: {d[k]:.4f}")

    g = load("results/graph_transition_huflit_results.json")
    print("\n=== HUFLIT Graph summary ===")
    for s in g["summary"]:
        print(s)


if __name__ == "__main__":
    main()
