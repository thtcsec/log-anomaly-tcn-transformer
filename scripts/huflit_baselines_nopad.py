"""Quick HUFLIT classical baselines only (no deep nets) after no-PAD fix."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from huflit_experiment import (
    SEEDS,
    drain_parse,
    group_by_ip_sequences,
    load_huflit_data,
    run_baselines,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "huflit_baselines_nopad.json"


def main():
    t0 = time.perf_counter()
    raw = load_huflit_data()
    events = drain_parse(raw)
    data = group_by_ip_sequences(events)
    print(
        f"sequences={len(data)} anom_rate={data['y'].mean():.4f} "
        f"mean_len={data['SeqLen'].mean():.2f}",
        flush=True,
    )
    rows = []
    for seed in SEEDS:
        part = run_baselines(data, seed)
        for r in part:
            print(f"seed={seed} {r['Method']} F1={r['F1']:.4f}", flush=True)
        rows.extend(part)
    df = pd.DataFrame(rows)
    agg = {}
    for method, g in df.groupby("Method"):
        agg[method] = {
            "F1_mean": float(g["F1"].mean()),
            "F1_std": float(g["F1"].std(ddof=1)),
            "Precision_mean": float(g["Precision"].mean()),
            "Recall_mean": float(g["Recall"].mean()),
        }
    payload = {
        "protocol": "no_PAD_in_EventId; IF=95th_pct(-decision_function)",
        "seeds": SEEDS,
        "rows": rows,
        "aggregated": agg,
        "seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
