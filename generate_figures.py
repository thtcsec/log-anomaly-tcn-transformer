"""Regenerate ROC/PR/score-distribution figures referenced by vnict2026.tex.

Uses seed 42 with the same baseline setup as the original notebook so the
figures match `tab_auc_ap` and `tab_threshold_sensitivity`.
"""

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
MAX_ROWS = 500000
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4

np.random.seed(SEED)


def main():
    print('Loading dataset...', flush=True)
    ds = load_dataset('logfit-project/HDFS_v1', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= MAX_ROWS:
            break
        rows.append({
            'content': row['content'],
            'BlockId': row['block_id'],
            'LineAnomaly': int(row['anomaly']),
        })
    raw_df = pd.DataFrame(rows)
    print(f'Loaded {len(raw_df)} rows', flush=True)

    print('Drain3 parsing...', flush=True)
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=config)
    parsed = []
    t0 = time.perf_counter()
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df)):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])
    print(f'Templates: {events["EventId"].nunique()} | {time.perf_counter()-t0:.2f}s', flush=True)

    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    print(f'Blocks: {len(data)} (normal={(data.y==0).sum()}, anomaly={(data.y==1).sum()})', flush=True)

    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=SEED, stratify=data['y']
    )
    y_true = test_df['y'].values

    vec_pca = CountVectorizer()
    X_train_all = vec_pca.fit_transform(train_df['text']).toarray()
    X_test_all = vec_pca.transform(test_df['text']).toarray()
    normal_mask = train_df['y'].values == 0
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_train_all[normal_mask])
    X_test_scaled = scaler.transform(X_test_all)
    n_components = max(1, min(20, X_normal.shape[1], X_normal.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(X_normal)
    X_test_recon = pca.inverse_transform(pca.transform(X_test_scaled))
    pca_score = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)

    vec_svd = CountVectorizer()
    X_train_sp = vec_svd.fit_transform(train_df['text'])
    X_test_sp = vec_svd.transform(test_df['text'])
    X_normal_sp = X_train_sp[normal_mask]
    n_components_svd = max(1, min(20, X_normal_sp.shape[1] - 1, X_normal_sp.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components_svd, random_state=SEED)
    svd.fit(X_normal_sp)
    Z = svd.transform(X_test_sp)
    Xh = svd.inverse_transform(Z)
    Xd = X_test_sp.toarray()
    svd_score = np.mean((Xd - Xh) ** 2, axis=1)

    vec_if = TfidfVectorizer()
    normal_train = train_df[train_df['y'] == 0]
    X_if_train = vec_if.fit_transform(normal_train['text'])
    X_if_test = vec_if.transform(test_df['text'])
    contamination = max(0.001, min(0.2, train_df['y'].mean()))
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1
    )
    iso.fit(X_if_train)
    if_score = -iso.decision_function(X_if_test)

    score_dict = {
        'PCA': pca_score,
        'TruncatedSVD': svd_score,
        'Isolation Forest': if_score,
    }

    print('Plotting ROC curves...', flush=True)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curves on HDFS test set (seed 42)')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig('roc_curves.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    print('Plotting PR curves...', flush=True)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    for name, scores in score_dict.items():
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec_curve, prec_curve, linewidth=2, label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curves on HDFS test set (seed 42)')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig('pr_curves.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    print('Plotting score distributions...', flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
    for ax, (name, scores) in zip(axes, score_dict.items()):
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        lo, hi = np.percentile(scores, [1, 99])
        normal_plot = np.clip(normal_scores, lo, hi)
        anomaly_plot = np.clip(anomaly_scores, lo, hi)
        ax.hist(normal_plot, bins=40, alpha=0.65, label='Normal')
        ax.hist(anomaly_plot, bins=40, alpha=0.75, label='Anomaly')
        ax.set_title(name)
        ax.set_xlabel('Anomaly score (clipped 1-99% for display)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('score_distributions.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    print('All figures saved.', flush=True)


if __name__ == '__main__':
    main()
