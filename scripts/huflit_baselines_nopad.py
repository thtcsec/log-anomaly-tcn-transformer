"""Quick HUFLIT classical baselines only (no deep nets) after no-PAD fix.

Uses client-IP disjoint split before W=10/stride=5 windowing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from huflit_experiment import (
    SEEDS,
    drain_parse,
    load_huflit_data,
    run_baselines,
    split_clients_then_window,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "huflit_baselines_nopad.json"


def main():
    t0 = time.perf_counter()
    raw = load_huflit_data()
    events = drain_parse(raw)
    print(
        f"lines={len(events)} clients={events['ip'].nunique()} "
        f"anom_line_rate={events['LineAnomaly'].mean():.4f}",
        flush=True,
    )
    rows = []
    for seed in SEEDS:
        train_df, test_df = split_clients_then_window(events, seed)
        print(
            f"seed={seed} train_seq={len(train_df)} test_seq={len(test_df)} "
            f"test_anom={test_df['y'].mean():.4f}",
            flush=True,
        )
        part = run_baselines(None, seed, train_df=train_df, test_df=test_df)
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
        "protocol": (
            "no_PAD_in_EventId; IF=95th_pct(-decision_function); "
            "client-IP disjoint split before W=10/stride=5 windowing; short walks retained"
        ),
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
