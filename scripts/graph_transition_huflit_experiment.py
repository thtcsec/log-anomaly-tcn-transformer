"""HUFLIT-only transition-graph experiment (no HuggingFace dependency)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from graph_transition_experiment import TransitionGraph, SEEDS, FEATURE_NAMES

OUT = Path(__file__).resolve().parents[1] / "results" / "graph_transition_huflit_results.json"


def load_huflit():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    default = root / "data" / "careerhub_20260604_095930" / "access.csv"
    csv_path = Path(os.environ.get("HUFLIT_CAREER_CSV", default))
    if not csv_path.exists():
        alt = root / "careerhub_20260604_095930" / "access.csv"
        csv_path = alt if alt.exists() else csv_path
    df = pd.read_csv(csv_path)
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="drain3-huflit"):
        msg = f"{row.method} {row.path}"
        result = miner.add_log_message(msg)
        rows.append(
            {
                "ip": row.ip,
                "timestamp": row.timestamp,
                "EventId": int(result["cluster_id"]),
                "LineAnomaly": int(row.suspicious_signals != "-"),
                "suspicious": str(row.suspicious_signals),
            }
        )
    events = pd.DataFrame(rows)
    window, step = 10, 5
    seqs = []
    for ip, group in events.groupby("ip"):
        group = group.sort_values("timestamp")
        eids = group["EventId"].tolist()
        anoms = group["LineAnomaly"].tolist()
        if len(eids) < window:
            seqs.append({"EventId": eids + [0] * (window - len(eids)), "y": int(any(anoms))})
            continue
        for i in range(0, len(eids) - window + 1, step):
            seqs.append(
                {
                    "EventId": eids[i : i + window],
                    "y": int(any(anoms[i : i + window])),
                }
            )
    data = pd.DataFrame(seqs)
    data["text"] = data["EventId"].apply(lambda xs: " ".join(f"E{x}" for x in xs))
    return data, events


def eval_methods(data: pd.DataFrame):
    rows = []
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            data, test_size=0.3, random_state=seed, stratify=data["y"]
        )
        y_test = test_df["y"].values
        normal_train = train_df[train_df["y"] == 0]

        # Classic IF on TF-IDF (bag-of-events) — reproduces paper baseline behaviour
        vec = TfidfVectorizer()
        Xtr = vec.fit_transform(normal_train["text"])
        Xte = vec.transform(test_df["text"])
        cont = max(0.001, min(0.2, float(train_df["y"].mean())))
        iso = IsolationForest(n_estimators=200, contamination=cont, random_state=seed, n_jobs=-1)
        iso.fit(Xtr)
        pred = (iso.predict(Xte) == -1).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append({"Method": "IF-TFIDF", "Seed": seed, "Precision": float(p), "Recall": float(r), "F1": float(f1)})

        graph = TransitionGraph().fit(normal_train["EventId"].tolist())
        X_train = np.vstack([graph.sequence_features(s) for s in train_df["EventId"]])
        X_test = np.vstack([graph.sequence_features(s) for s in test_df["EventId"]])
        y_train = train_df["y"].values
        normal_mask = y_train == 0

        thr = np.percentile(X_train[normal_mask, 0], 95)
        pred = (X_test[:, 0] > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append(
            {
                "Method": "GraphWalk-NLL",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": len(graph.vocab),
                "Edges": int(sum(len(c) for c in graph.edge_counts.values())),
            }
        )

        scaler = StandardScaler()
        Xn = scaler.fit_transform(X_train[normal_mask])
        Xt = scaler.transform(X_test)
        n_comp = max(1, min(4, Xn.shape[1], Xn.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=seed).fit(Xn)
        train_err = np.mean((Xn - pca.inverse_transform(pca.transform(Xn))) ** 2, axis=1)
        thr = np.percentile(train_err, 95)
        test_err = np.mean((Xt - pca.inverse_transform(pca.transform(Xt))) ** 2, axis=1)
        pred = (test_err > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append({"Method": "PCA+TransFeat", "Seed": seed, "Precision": float(p), "Recall": float(r), "F1": float(f1)})

        iso2 = IsolationForest(n_estimators=200, contamination=cont, random_state=seed, n_jobs=-1)
        iso2.fit(X_train[normal_mask])
        train_scores = -iso2.decision_function(X_train[normal_mask])
        thr = np.percentile(train_scores, 95)
        scores = -iso2.decision_function(X_test)
        pred = (scores > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append({"Method": "IF+TransFeat", "Seed": seed, "Precision": float(p), "Recall": float(r), "F1": float(f1)})

        print(
            f"seed={seed} IF-TFIDF={rows[-4]['F1']:.4f} GraphWalk={rows[-3]['F1']:.4f} "
            f"PCA+TF={rows[-2]['F1']:.4f} IF+TF={rows[-1]['F1']:.4f}",
            flush=True,
        )
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    for method, g in df.groupby("Method"):
        out.append(
            {
                "Method": method,
                "F1_mean": float(g["F1"].mean()),
                "F1_std": float(g["F1"].std(ddof=1)),
                "P_mean": float(g["Precision"].mean()),
                "R_mean": float(g["Recall"].mean()),
            }
        )
    return out


def main():
    t0 = time.perf_counter()
    data, events = load_huflit()
    print(
        f"sequences={len(data)} anomaly_rate={data.y.mean():.4f} templates={events.EventId.nunique()}",
        flush=True,
    )
    # Label rule diagnostics for paper limitation section
    label_diag = {
        "rows": int(len(events)),
        "anomaly_lines": int(events.LineAnomaly.sum()),
        "unique_suspicious_tokens": sorted(
            {
                tok.strip()
                for s in events.loc[events.LineAnomaly == 1, "suspicious"].unique()
                for tok in str(s).replace("|", ",").split(",")
                if tok.strip() and tok.strip() != "-"
            }
        )[:50],
        "note": "Labels come from fixed suspicious_signals rules in the access log export.",
    }
    rows = eval_methods(data)
    payload = {
        "seeds": SEEDS,
        "feature_names": FEATURE_NAMES,
        "label_diagnostics": label_diag,
        "rows": rows,
        "summary": summarize(rows),
        "elapsed_sec": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
