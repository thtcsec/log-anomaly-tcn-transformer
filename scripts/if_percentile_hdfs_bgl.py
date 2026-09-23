"""Rerun Isolation Forest on HDFS/BGL with unified 95th-percentile protocol.

Fit on normal TF-IDF only; threshold -decision_function at the 95th percentile
of normal training scores (same family as PCA / GraphWalk / HUFLIT headline IF).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEEDS = [21, 42, 84, 123, 777]
MAX_ROWS = 500_000
OUT = Path("logs/if_percentile_hdfs_bgl.json")


def stream_rows(name: str, max_rows: int):
    from datasets import load_dataset

    ds = load_dataset(f"logfit-project/{name}", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        if name == "HDFS_v1":
            rows.append(
                {
                    "content": row["content"],
                    "BlockId": row["block_id"],
                    "LineAnomaly": int(row["anomaly"]),
                }
            )
        else:
            rows.append(
                {
                    "content": row["content"],
                    "LineAnomaly": int(row["anomaly"]),
                }
            )
    return pd.DataFrame(rows)


def drain_parse(contents, labels):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    event_ids, anoms = [], []
    for content, y in tqdm(zip(contents, labels), total=len(contents), desc="drain3"):
        result = miner.add_log_message(str(content))
        event_ids.append(int(result["cluster_id"]))
        anoms.append(int(y))
    return pd.DataFrame({"EventId": event_ids, "LineAnomaly": anoms})


def hdfs_sequences(raw: pd.DataFrame) -> pd.DataFrame:
    events = drain_parse(raw["content"], raw["LineAnomaly"])
    events["BlockId"] = raw["BlockId"].values
    return (
        events.groupby("BlockId")
        .agg(EventId=("EventId", list), y=("LineAnomaly", "max"))
        .reset_index()
    )


def bgl_sequences(raw: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    events = drain_parse(raw["content"], raw["LineAnomaly"])
    n = len(events)
    seqs, ys = [], []
    for start in range(0, n - window + 1, window):
        chunk = events.iloc[start : start + window]
        seqs.append(chunk["EventId"].tolist())
        ys.append(int(chunk["LineAnomaly"].max()))
    return pd.DataFrame({"EventId": seqs, "y": ys})


def run_if(data: pd.DataFrame, seed: int) -> dict:
    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=seed, stratify=data["y"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["text"] = train_df["EventId"].apply(lambda s: " ".join(map(str, s)))
    test_df["text"] = test_df["EventId"].apply(lambda s: " ".join(map(str, s)))

    normal_train = train_df[train_df["y"] == 0]
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(normal_train["text"])
    X_test = vec.transform(test_df["text"])

    iso = IsolationForest(
        n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
    )
    iso.fit(X_train)
    train_scores = -iso.decision_function(X_train)
    thr = float(np.percentile(train_scores, 95))
    test_scores = -iso.decision_function(X_test)
    y_pred = (test_scores > thr).astype(int)
    y_test = test_df["y"].astype(int).values
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    return {
        "Seed": seed,
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "threshold": thr,
    }


def summarize(rows):
    f1s = [r["F1"] for r in rows]
    return {
        "mean": float(np.mean(f1s)),
        "std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "per_seed": rows,
    }


def main():
    t0 = time.perf_counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("=== HDFS 500k IF percentile ===", flush=True)
    hdfs_raw = stream_rows("HDFS_v1", MAX_ROWS)
    hdfs_data = hdfs_sequences(hdfs_raw)
    print(f"HDFS sequences={len(hdfs_data)} anom={hdfs_data['y'].mean():.4f}", flush=True)
    hdfs_rows = []
    for seed in SEEDS:
        row = run_if(hdfs_data, seed)
        print(f"  seed={seed} F1={row['F1']:.4f}", flush=True)
        hdfs_rows.append(row)

    print("=== BGL 500k IF percentile ===", flush=True)
    bgl_raw = stream_rows("BGL", MAX_ROWS)
    bgl_data = bgl_sequences(bgl_raw)
    print(f"BGL sequences={len(bgl_data)} anom={bgl_data['y'].mean():.4f}", flush=True)
    bgl_rows = []
    for seed in SEEDS:
        row = run_if(bgl_data, seed)
        print(f"  seed={seed} F1={row['F1']:.4f}", flush=True)
        bgl_rows.append(row)

    payload = {
        "protocol": "fit_normal_only; threshold=95th_pct(-decision_function)",
        "max_rows": MAX_ROWS,
        "seeds": SEEDS,
        "HDFS": summarize(hdfs_rows),
        "BGL": summarize(bgl_rows),
        "seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"HDFS IF F1={payload['HDFS']['mean']:.4f}±{payload['HDFS']['std']:.4f}",
        flush=True,
    )
    print(
        f"BGL  IF F1={payload['BGL']['mean']:.4f}±{payload['BGL']['std']:.4f}",
        flush=True,
    )
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
