"""HUFLIT ablation: Transition-only vs frequency IF vs late-fusion (z-score avg).

Lightweight (no deep nets): validates that transition-graph scores enter the detector.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from graph_transition_experiment import FEATURE_NAMES, SEEDS, TransitionGraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "huflit_graph_ablation.json"


def load_sequences():
    default = ROOT / "data" / "careerhub_20260604_095930" / "access.csv"
    csv_path = Path(os.environ.get("HUFLIT_CAREER_CSV", default))
    df = pd.read_csv(csv_path)
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="drain3"):
        msg = f"{row.method} {row.path}"
        result = miner.add_log_message(msg)
        rows.append(
            {
                "ip": row.ip,
                "timestamp": row.timestamp,
                "EventId": int(result["cluster_id"]),
                "y_line": int(row.suspicious_signals != "-"),
            }
        )
    events = pd.DataFrame(rows)
    seqs = []
    window, step = 10, 5
    for ip, g in events.groupby("ip"):
        g = g.sort_values("timestamp")
        eids, ys = g["EventId"].tolist(), g["y_line"].tolist()
        if len(eids) < window:
            seqs.append({"EventId": eids + [0] * (window - len(eids)), "y": int(any(ys))})
            continue
        for i in range(0, len(eids) - window + 1, step):
            seqs.append(
                {
                    "EventId": eids[i : i + window],
                    "y": int(any(ys[i : i + window])),
                }
            )
    data = pd.DataFrame(seqs)
    data["text"] = data["EventId"].apply(lambda xs: " ".join(f"E{x}" for x in xs))
    return data


def zscore(train_scores, test_scores):
    mu, sd = float(np.mean(train_scores)), float(np.std(train_scores) + 1e-8)
    return (test_scores - mu) / sd, (train_scores - mu) / sd


def eval_seed(data, seed):
    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=seed, stratify=data["y"]
    )
    y_test = test_df["y"].values
    normal_train = train_df[train_df["y"] == 0]
    graph = TransitionGraph().fit(normal_train["EventId"].tolist())

    tr_nll = np.array([graph.walk_nll(s) for s in train_df["EventId"]], dtype=np.float64)
    te_nll = np.array([graph.walk_nll(s) for s in test_df["EventId"]], dtype=np.float64)
    thr = np.percentile(tr_nll[train_df["y"].values == 0], 95)
    pred_g = (te_nll > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred_g, average="binary", zero_division=0)
    auc_g = roc_auc_score(y_test, te_nll) if len(np.unique(y_test)) > 1 else float("nan")

    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(normal_train["text"])
    Xte = vec.transform(test_df["text"])
    cont = max(0.001, min(0.2, float(train_df["y"].mean())))
    iso = IsolationForest(n_estimators=200, contamination=cont, random_state=seed, n_jobs=-1)
    iso.fit(Xtr)
    # higher anomaly score = more anomalous
    te_if = -iso.decision_function(Xte)
    tr_if = -iso.decision_function(vec.transform(train_df["text"]))
    thr_if = np.percentile(tr_if[train_df["y"].values == 0], 95)
    pred_if = (te_if > thr_if).astype(int)
    p_i, r_i, f1_i, _ = precision_recall_fscore_support(
        y_test, pred_if, average="binary", zero_division=0
    )
    auc_i = roc_auc_score(y_test, te_if)

    # Late fusion: z-score average of GraphWalk NLL and IF score (proxy for TCN+graph when DL unavailable)
    te_gz, tr_gz = zscore(tr_nll[train_df["y"].values == 0], te_nll)
    # recompute z using normal-only train refs for IF
    te_iz, _ = zscore(tr_if[train_df["y"].values == 0], te_if)
    te_fuse = 0.5 * te_gz + 0.5 * te_iz
    tr_fuse_normal = 0.5 * ((tr_nll[train_df["y"].values == 0] - np.mean(tr_nll[train_df["y"].values == 0])) / (np.std(tr_nll[train_df["y"].values == 0]) + 1e-8))
    # approximate fuse threshold on normal train: rebuild
    tr_n_nll = tr_nll[train_df["y"].values == 0]
    tr_n_if = tr_if[train_df["y"].values == 0]
    tr_fuse = 0.5 * ((tr_n_nll - tr_n_nll.mean()) / (tr_n_nll.std() + 1e-8)) + 0.5 * (
        (tr_n_if - tr_n_if.mean()) / (tr_n_if.std() + 1e-8)
    )
    thr_f = np.percentile(tr_fuse, 95)
    pred_f = (te_fuse > thr_f).astype(int)
    p_f, r_f, f1_f, _ = precision_recall_fscore_support(
        y_test, pred_f, average="binary", zero_division=0
    )
    auc_f = roc_auc_score(y_test, te_fuse)

    # sparsity diagnostic
    nnz = Xte.nnz / max(Xte.shape[0] * Xte.shape[1], 1)

    return {
        "Seed": seed,
        "GraphWalk_F1": float(f1),
        "GraphWalk_AUC": float(auc_g),
        "IF_TFIDF_F1": float(f1_i),
        "IF_TFIDF_AUC": float(auc_i),
        "Fusion_Graph_IF_F1": float(f1_f),
        "Fusion_Graph_IF_AUC": float(auc_f),
        "TFIDF_density": float(nnz),
        "contamination": float(cont),
        "templates": int(len(graph.vocab)),
    }


def main():
    data = load_sequences()
    rows = [eval_seed(data, s) for s in SEEDS]
    df = pd.DataFrame(rows)
    summary = {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1))}
        for c in df.columns
        if c != "Seed"
    }
    payload = {"seeds": SEEDS, "feature_names": FEATURE_NAMES, "rows": rows, "summary": summary}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
