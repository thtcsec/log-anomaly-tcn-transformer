"""Transition-graph feature experiment for CSoNet camera-ready.

Builds a directed event-transition graph from normal training sequences and
evaluates (i) GraphWalk NLL scoring and (ii) PCA / Isolation Forest on
transition-probability features — making the network-aware framing concrete.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SEEDS = [21, 42, 84, 123, 777]
OUT = Path(__file__).resolve().parents[1] / "logs" / "graph_transition_results.json"


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def drain_parse_contents(contents, labels):
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


def load_hdfs(max_rows=200_000):
    ds = _load_dataset("logfit-project/HDFS_v1", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append(
            {
                "content": row["content"],
                "BlockId": row["block_id"],
                "LineAnomaly": int(row["anomaly"]),
            }
        )
    raw = pd.DataFrame(rows)
    events = drain_parse_contents(raw["content"], raw["LineAnomaly"])
    events["BlockId"] = raw["BlockId"].values
    data = (
        events.groupby("BlockId")
        .agg(EventId=("EventId", list), y=("LineAnomaly", "max"))
        .reset_index()
    )
    return data, "HDFS"


def load_bgl(max_rows=200_000, window=100):
    ds = _load_dataset("logfit-project/BGL", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append(
            {
                "content": row.get("content", row.get("Content", "")),
                "LineAnomaly": int(row.get("anomaly", 0)),
            }
        )
    raw = pd.DataFrame(rows)
    events = drain_parse_contents(raw["content"], raw["LineAnomaly"])
    seqs = []
    for start in range(0, len(events) - window + 1, window):
        chunk = events.iloc[start : start + window]
        seqs.append(
            {
                "EventId": chunk["EventId"].tolist(),
                "y": int(chunk["LineAnomaly"].max()),
            }
        )
    return pd.DataFrame(seqs), "BGL"


def window_huflit_events(events: pd.DataFrame, window: int = 10, step: int = 5) -> pd.DataFrame:
    """Build W/stride walks per client IP (short clients kept as variable-length walks)."""
    rows = []
    for ip, group in events.groupby("ip"):
        group = group.sort_values("timestamp")
        eids = group["EventId"].tolist()
        anoms = group["LineAnomaly"].tolist()
        if len(eids) < window:
            rows.append({"EventId": eids, "y": int(any(anoms)), "ip": ip})
            continue
        for i in range(0, len(eids) - window + 1, step):
            rows.append(
                {
                    "EventId": eids[i : i + window],
                    "y": int(any(anoms[i : i + window])),
                    "ip": ip,
                }
            )
    data = pd.DataFrame(rows)
    data["text"] = data["EventId"].apply(
        lambda xs: " ".join(f"E{x}" for x in xs if int(x) != 0)
    )
    return data


def load_huflit_events():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    default = root / "data" / "careerhub_20260604_095930" / "access.csv"
    csv_path = Path(os.environ.get("HUFLIT_CAREER_CSV", default))
    if not csv_path.exists():
        alt = root / "careerhub_20260604_095930" / "access.csv"
        csv_path = alt if alt.exists() else csv_path
    df = pd.read_csv(csv_path)
    contents = (df["method"].astype(str) + " " + df["path"].astype(str)).tolist()
    labels = (df["suspicious_signals"] != "-").astype(int).tolist()
    events = drain_parse_contents(contents, labels)
    events["ip"] = df["ip"].values
    events["timestamp"] = df["timestamp"].values
    return events, "HUFLIT-Career"


def split_huflit_by_client(events: pd.DataFrame, seed: int, test_size: float = 0.3):
    """Split clients first, then window — no raw-request overlap across partitions."""
    ip_y = events.groupby("ip")["LineAnomaly"].max().reset_index()
    ip_ids = ip_y["ip"].astype(str).to_numpy()
    labels = ip_y["LineAnomaly"].astype(int).to_numpy()
    train_ips, test_ips = train_test_split(
        ip_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_set, test_set = set(train_ips), set(test_ips)
    train_df = window_huflit_events(events[events["ip"].astype(str).isin(train_set)])
    test_df = window_huflit_events(events[events["ip"].astype(str).isin(test_set)])
    return train_df, test_df


def load_huflit():
    """Legacy helper: window all clients (prefer client-disjoint eval_huflit path)."""
    events, name = load_huflit_events()
    return window_huflit_events(events), name


class TransitionGraph:
    """Directed first-order Markov transition graph over event templates."""

    def __init__(self, alpha: float = 1.0, pad_id: int = 0):
        self.alpha = alpha
        self.pad_id = pad_id
        self.edge_counts = defaultdict(Counter)
        self.node_out = Counter()
        self.node_visit = Counter()
        self.vocab = set()

    @staticmethod
    def clean_seq(seq, pad_id: int = 0):
        """Drop padding tokens; GraphWalk vertices are Drain3 event IDs only."""
        return [int(e) for e in seq if int(e) != pad_id]

    def fit(self, sequences):
        for seq in sequences:
            seq = self.clean_seq(seq, self.pad_id)
            if len(seq) < 1:
                continue
            self.node_visit[seq[0]] += 1
            self.vocab.add(seq[0])
            for a, b in zip(seq[:-1], seq[1:]):
                self.edge_counts[a][b] += 1
                self.node_out[a] += 1
                self.node_visit[b] += 1
                self.vocab.add(a)
                self.vocab.add(b)
        return self

    def _support_size(self) -> int:
        # Observed training nodes V plus one UNK bucket for novel destinations.
        return max(len(self.vocab), 1) + 1

    def transition_prob(self, a, b):
        """Laplace-smoothed P(b|a) over V ∪ {UNK}.

        Novel destinations b∉V map to UNK (count 0). Unseen sources use uniform 1/(|V|+1).
        """
        v_unk = self._support_size()
        if a not in self.vocab or self.node_out[a] == 0:
            return 1.0 / v_unk
        count_ab = 0 if b not in self.vocab else self.edge_counts[a][b]
        return (count_ab + self.alpha) / (self.node_out[a] + self.alpha * v_unk)

    def sequence_features(self, seq):
        seq = self.clean_seq(seq, self.pad_id)
        if len(seq) < 2:
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0, float(len(seq))], dtype=np.float64)
        logps = []
        unseen = 0
        novel_nodes = 0
        for a, b in zip(seq[:-1], seq[1:]):
            if a not in self.vocab or b not in self.vocab or self.edge_counts[a][b] == 0:
                unseen += 1
            if a not in self.vocab:
                novel_nodes += 1
            logps.append(np.log(max(self.transition_prob(a, b), 1e-12)))
        if seq[-1] not in self.vocab:
            novel_nodes += 1
        logps = np.asarray(logps, dtype=np.float64)
        n_edge = max(len(seq) - 1, 1)
        return np.array(
            [
                float(-logps.mean()),  # mean NLL of walk
                float(-logps.min()),  # worst-edge surprise
                float(unseen / n_edge),  # rare/unseen transition ratio
                float(novel_nodes / max(len(seq), 1)),
                float(len(set(seq)) / max(len(seq), 1)),  # unique-event ratio
                float(len(seq)),
            ],
            dtype=np.float64,
        )

    def walk_nll(self, seq):
        return float(self.sequence_features(seq)[0])


FEATURE_NAMES = [
    "mean_nll",
    "max_edge_nll",
    "unseen_edge_ratio",
    "novel_node_ratio",
    "unique_ratio",
    "seq_len",
]


def eval_dataset(data: pd.DataFrame, name: str):
    rows = []
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            data, test_size=0.3, random_state=seed, stratify=data["y"]
        )
        normal_train = train_df[train_df["y"] == 0]["EventId"].tolist()
        graph = TransitionGraph(alpha=1.0).fit(normal_train)

        X_train = np.vstack([graph.sequence_features(s) for s in train_df["EventId"]])
        X_test = np.vstack([graph.sequence_features(s) for s in test_df["EventId"]])
        y_train = train_df["y"].values
        y_test = test_df["y"].values
        normal_mask = y_train == 0

        # GraphWalk: threshold mean NLL on normal validation split of train
        nll_train = X_train[normal_mask, 0]
        thr = np.percentile(nll_train, 95)
        y_pred = (X_test[:, 0] > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Dataset": name,
                "Method": "GraphWalk-NLL",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": len(graph.vocab),
                "Edges": int(sum(len(c) for c in graph.edge_counts.values())),
            }
        )

        # PCA on transition features
        scaler = StandardScaler()
        Xn = scaler.fit_transform(X_train[normal_mask])
        Xt = scaler.transform(X_test)
        n_comp = max(1, min(4, Xn.shape[1], Xn.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=seed)
        pca.fit(Xn)
        train_err = np.mean((Xn - pca.inverse_transform(pca.transform(Xn))) ** 2, axis=1)
        thr = np.percentile(train_err, 95)
        test_err = np.mean((Xt - pca.inverse_transform(pca.transform(Xt))) ** 2, axis=1)
        y_pred = (test_err > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Dataset": name,
                "Method": "PCA+TransFeat",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": len(graph.vocab),
                "Edges": int(sum(len(c) for c in graph.edge_counts.values())),
            }
        )

        # Isolation Forest on transition features (95th-pct protocol; no label-informed contamination)
        iso = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
        )
        iso.fit(X_train[normal_mask])
        # decision_function: higher = more normal; use negative as anomaly score
        scores = -iso.decision_function(X_test)
        # threshold from normal train scores
        train_scores = -iso.decision_function(X_train[normal_mask])
        thr = np.percentile(train_scores, 95)
        y_pred = (scores > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Dataset": name,
                "Method": "IF+TransFeat",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": len(graph.vocab),
                "Edges": int(sum(len(c) for c in graph.edge_counts.values())),
            }
        )
        print(
            f"[{name} seed={seed}] GraphWalk={rows[-3]['F1']:.4f} "
            f"PCA+TF={rows[-2]['F1']:.4f} IF+TF={rows[-1]['F1']:.4f}",
            flush=True,
        )
    return rows


def eval_huflit_client_split(events: pd.DataFrame, name: str = "HUFLIT-Career"):
    """Per-seed client-disjoint split, then GraphWalk / transition-feature baselines."""
    rows = []
    for seed in SEEDS:
        train_df, test_df = split_huflit_by_client(events, seed)
        # Reuse the same scoring block as eval_dataset via a temporary frame API:
        # inline to keep seed-specific train/test (not a global window pool).
        normal_train = train_df[train_df["y"] == 0]["EventId"].tolist()
        graph = TransitionGraph(alpha=1.0).fit(normal_train)
        X_train = np.vstack([graph.sequence_features(s) for s in train_df["EventId"]])
        X_test = np.vstack([graph.sequence_features(s) for s in test_df["EventId"]])
        y_train = train_df["y"].values
        y_test = test_df["y"].values
        normal_mask = y_train == 0

        thr = np.percentile(X_train[normal_mask, 0], 95)
        y_pred = (X_test[:, 0] > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        nodes = len(graph.vocab)
        edges = int(sum(len(c) for c in graph.edge_counts.values()))
        rows.append(
            {
                "Dataset": name,
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
        y_pred = (test_err > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Dataset": name,
                "Method": "PCA+TransFeat",
                "Seed": seed,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f1),
                "Nodes": nodes,
                "Edges": edges,
            }
        )

        iso = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
        )
        iso.fit(X_train[normal_mask])
        train_scores = -iso.decision_function(X_train[normal_mask])
        thr = np.percentile(train_scores, 95)
        scores = -iso.decision_function(X_test)
        y_pred = (scores > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Dataset": name,
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
            f"[{name} seed={seed}] GraphWalk={rows[-3]['F1']:.4f} "
            f"PCA+TF={rows[-2]['F1']:.4f} IF+TF={rows[-1]['F1']:.4f}",
            flush=True,
        )
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    summary = []
    for (dataset, method), g in df.groupby(["Dataset", "Method"]):
        summary.append(
            {
                "Dataset": dataset,
                "Method": method,
                "F1_mean": float(g["F1"].mean()),
                "F1_std": float(g["F1"].std(ddof=1)),
                "P_mean": float(g["Precision"].mean()),
                "R_mean": float(g["Recall"].mean()),
                "Nodes": int(g["Nodes"].iloc[0]),
                "Edges": int(g["Edges"].iloc[0]),
            }
        )
    return summary


def main():
    t0 = time.perf_counter()
    all_rows = []
    for loader in (load_hdfs, load_bgl):
        data, name = loader()
        print(f"\n=== {name}: {len(data)} sequences, anomaly rate={data.y.mean():.4f} ===", flush=True)
        all_rows.extend(eval_dataset(data, name))
    events, name = load_huflit_events()
    print(
        f"\n=== {name}: client-disjoint split then W=10/stride=5 "
        f"({events.ip.nunique()} clients) ===",
        flush=True,
    )
    all_rows.extend(eval_huflit_client_split(events, name))
    summary = summarize(all_rows)
    payload = {
        "seeds": SEEDS,
        "feature_names": FEATURE_NAMES,
        "protocol_notes": {
            "HUFLIT": "client-IP disjoint split before windowing (W=10, stride=5); short walks retained",
            "GraphWalk": "Laplace over V∪{UNK}; novel destinations map to UNK",
            "HDFS_BGL": "sequence-level split (Block ID / non-overlapping W=100); no sliding overlap",
        },
        "rows": all_rows,
        "summary": summary,
        "elapsed_sec": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== Summary ===", flush=True)
    for s in summary:
        print(
            f"{s['Dataset']:<14} {s['Method']:<14} "
            f"F1={s['F1_mean']:.4f}±{s['F1_std']:.4f} "
            f"(nodes={s['Nodes']}, edges={s['Edges']})",
            flush=True,
        )
    print(f"Saved {OUT} in {payload['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
