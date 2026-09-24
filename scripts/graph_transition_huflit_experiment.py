"""HUFLIT-only GraphWalk / transition-feature experiment (no HuggingFace).

Protocol matches the paper headline: client-IP disjoint split *before*
W=10/stride=5 windowing; GraphWalk Laplace over V∪{UNK}.
For the full HDFS/BGL/HUFLIT dump prefer scripts/graph_transition_experiment.py.
"""

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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from graph_transition_experiment import (
    FEATURE_NAMES,
    SEEDS,
    TransitionGraph,
    load_huflit_events,
    split_huflit_by_client,
)

OUT = Path(__file__).resolve().parents[1] / "logs" / "graph_transition_huflit_results.json"


def eval_methods(events: pd.DataFrame):
    rows = []
    for seed in SEEDS:
        train_df, test_df = split_huflit_by_client(events, seed)
        y_test = test_df["y"].values
        normal_train = train_df[train_df["y"] == 0]

        vec = TfidfVectorizer()
        Xtr = vec.fit_transform(normal_train["text"])
        Xte = vec.transform(test_df["text"])
        iso = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
        )
        iso.fit(Xtr)
        tr_scores = -iso.decision_function(Xtr)
        thr = np.percentile(tr_scores, 95)
        pred = (-iso.decision_function(Xte) > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Method": "IF-TFIDF",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
            }
        )

        graph = TransitionGraph().fit(normal_train["EventId"].tolist())
        X_train = np.vstack([graph.sequence_features(s) for s in train_df["EventId"]])
        X_test = np.vstack([graph.sequence_features(s) for s in test_df["EventId"]])
        y_train = train_df["y"].values
        normal_mask = y_train == 0
        nodes = len(graph.vocab)
        edges = int(sum(len(c) for c in graph.edge_counts.values()))

        thr = np.percentile(X_train[normal_mask, 0], 95)
        pred = (X_test[:, 0] > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Method": "GraphWalk-NLL",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": nodes,
                "Edges": edges,
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
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Method": "PCA+TransFeat",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": nodes,
                "Edges": edges,
            }
        )

        iso2 = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
        )
        iso2.fit(X_train[normal_mask])
        train_scores = -iso2.decision_function(X_train[normal_mask])
        thr = np.percentile(train_scores, 95)
        scores = -iso2.decision_function(X_test)
        pred = (scores > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Method": "IF+TransFeat",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": nodes,
                "Edges": edges,
            }
        )

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
    events, name = load_huflit_events()
    print(
        f"{name}: {len(events)} lines, {events['ip'].nunique()} clients "
        f"(client-disjoint → W=10/stride=5)",
        flush=True,
    )
    label_diag = {
        "rows": int(len(events)),
        "anomaly_lines": int(events["LineAnomaly"].sum()),
        "note": "Labels come from fixed suspicious_signals rules in the access log export.",
        "split": "client-IP disjoint before windowing",
    }
    rows = eval_methods(events)
    payload = {
        "seeds": SEEDS,
        "feature_names": FEATURE_NAMES,
        "protocol": "client-IP disjoint; W=10/stride=5; GraphWalk V∪{UNK}; IF 95th-pct",
        "label_diagnostics": label_diag,
        "rows": rows,
        "summary": summarize(rows),
        "elapsed_sec": time.perf_counter() - t0,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
