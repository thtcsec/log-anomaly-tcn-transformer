"""Redraw sharper ROC/PR curves from cached baseline scores if available,
or regenerate baselines-only high-DPI figures on HDFS (fast path).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SEED = 42
MAX_ROWS = 200_000
OUT_DIR = Path("images")


def try_load_hf(max_rows: int):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("datasets not installed") from e
    ds = load_dataset("logfit-project/HDFS_v1", split="train", streaming=True)
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
    return pd.DataFrame(rows)


def parse_and_group(raw: pd.DataFrame):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    parsed = []
    for row in tqdm(raw.itertuples(index=False), total=len(raw), desc="drain3"):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result["cluster_id"])))
    events = pd.DataFrame(parsed, columns=["BlockId", "LineAnomaly", "EventId"])
    data = (
        events.groupby("BlockId")
        .agg(EventId=("EventId", list), y=("LineAnomaly", "max"))
        .reset_index()
    )
    data["text"] = data["EventId"].apply(lambda xs: " ".join(f"E{x}" for x in xs))
    return data


def baseline_scores(train_df, test_df):
    normal_mask = train_df["y"].values == 0
    scores = {}

    vec = CountVectorizer()
    Xtr = vec.fit_transform(train_df["text"]).toarray()
    Xte = vec.transform(test_df["text"]).toarray()
    scaler = StandardScaler()
    Xn = scaler.fit_transform(Xtr[normal_mask])
    Xt = scaler.transform(Xte)
    n_comp = max(1, min(20, Xn.shape[1], Xn.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=SEED).fit(Xn)
    scores["PCA"] = np.mean((Xt - pca.inverse_transform(pca.transform(Xt))) ** 2, axis=1)

    vec2 = CountVectorizer()
    Xtr_sp = vec2.fit_transform(train_df["text"])
    Xte_sp = vec2.transform(test_df["text"])
    Xn_sp = Xtr_sp[normal_mask]
    n_svd = max(1, min(20, Xn_sp.shape[1] - 1, Xn_sp.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_svd, random_state=SEED).fit(Xn_sp)
    Xh = svd.inverse_transform(svd.transform(Xte_sp))
    scores["TruncatedSVD"] = np.mean((Xte_sp.toarray() - Xh) ** 2, axis=1)

    vec_if = TfidfVectorizer()
    X_if_tr = vec_if.fit_transform(train_df[train_df["y"] == 0]["text"])
    X_if_te = vec_if.transform(test_df["text"])
    contamination = max(0.001, min(0.2, float(train_df["y"].mean())))
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1
    ).fit(X_if_tr)
    scores["Isolation Forest"] = -iso.decision_function(X_if_te)
    return scores


COLORS = {
    "PCA": "#0072B2",
    "TruncatedSVD": "#E69F00",
    "Isolation Forest": "#009E73",
    "GraphWalk-NLL": "#D55E00",
    "DeepLog (NLL)": "#CC79A7",
    "TCN (NLL)": "#56B4E9",
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def plot_roc(score_dict, y_true, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=120)
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=2.4,
            color=COLORS.get(name, "#333333"),
            label=f"{name} (AUC={roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC on HDFS (seed 42)")
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def plot_pr(score_dict, y_true, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=120)
    for name, scores in score_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(
            rec,
            prec,
            lw=2.4,
            color=COLORS.get(name, "#333333"),
            label=f"{name} (AP={ap:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall on HDFS (seed 42)")
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    raw = try_load_hf(MAX_ROWS)
    data = parse_and_group(raw)
    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=SEED, stratify=data["y"]
    )
    scores = baseline_scores(train_df, test_df)

    # Optional: add GraphWalk if module available
    try:
        from graph_transition_experiment import TransitionGraph

        graph = TransitionGraph().fit(train_df[train_df["y"] == 0]["EventId"].tolist())
        scores["GraphWalk-NLL"] = np.array(
            [graph.walk_nll(s) for s in test_df["EventId"]], dtype=np.float64
        )
    except Exception as e:
        print(f"GraphWalk skip: {e}", flush=True)

    meta = {
        name: {
            "auc": float(auc(*roc_curve(test_df["y"].values, s)[:2])),
            "ap": float(average_precision_score(test_df["y"].values, s)),
        }
        for name, s in scores.items()
    }
    Path("figure_curve_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_roc(scores, test_df["y"].values, OUT_DIR / "roc_curves.png")
    plot_pr(scores, test_df["y"].values, OUT_DIR / "pr_curves.png")


if __name__ == "__main__":
    main()
