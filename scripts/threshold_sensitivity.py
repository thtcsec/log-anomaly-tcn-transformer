"""Recompute threshold-sensitivity and AUC/AP for seed 42, matching the
multi-seed pipeline so the numbers in tab_threshold_sensitivity and
tab_auc_ap are consistent with tab_results."""

import json
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

SEED = 42
MAX_ROWS = 500000


def main():
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

    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    parsed = []
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc='drain3'):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])

    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))

    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=SEED, stratify=data['y']
    )
    y_test = test_df['y'].values

    # PCA
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
    X_train_recon = pca.inverse_transform(pca.transform(X_normal))
    pca_train_err = np.mean((X_normal - X_train_recon) ** 2, axis=1)
    X_test_recon = pca.inverse_transform(pca.transform(X_test_scaled))
    pca_test_err = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)

    # SVD
    vec_svd = CountVectorizer()
    X_train_sp = vec_svd.fit_transform(train_df['text'])
    X_test_sp = vec_svd.transform(test_df['text'])
    X_normal_sp = X_train_sp[normal_mask]
    n_components_svd = max(1, min(20, X_normal_sp.shape[1] - 1, X_normal_sp.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components_svd, random_state=SEED)
    svd.fit(X_normal_sp)

    def svd_err(X):
        Z = svd.transform(X)
        Xh = svd.inverse_transform(Z)
        Xd = X.toarray()
        return np.mean((Xd - Xh) ** 2, axis=1)

    svd_train_err = svd_err(X_normal_sp)
    svd_test_err = svd_err(X_test_sp)

    # IF
    vec_if = TfidfVectorizer()
    normal_train = train_df[train_df['y'] == 0]
    X_if_train = vec_if.fit_transform(normal_train['text'])
    X_if_test = vec_if.transform(test_df['text'])
    contamination = max(0.001, min(0.2, train_df['y'].mean()))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1)
    iso.fit(X_if_train)
    if_train_scores = -iso.decision_function(X_if_train)
    if_test_scores = -iso.decision_function(X_if_test)

    score_dict = {
        'PCA': (pca_train_err, pca_test_err),
        'TruncatedSVD': (svd_train_err, svd_test_err),
        'Isolation Forest': (if_train_scores, if_test_scores),
    }

    auc_rows = []
    for name, (_, test_scores) in score_dict.items():
        roc_auc = roc_auc_score(y_test, test_scores)
        ap = average_precision_score(y_test, test_scores)
        auc_rows.append({'Method': name, 'ROC_AUC': roc_auc, 'AP': ap})
    print('=== AUC/AP (seed 42) ===')
    for r in auc_rows:
        print(f"  {r['Method']:<20} ROC_AUC={r['ROC_AUC']:.4f}  AP={r['AP']:.4f}")

    percentiles = [90, 95, 97, 99]
    sens_rows = []
    for name, (train_scores, test_scores) in score_dict.items():
        for perc in percentiles:
            th = np.percentile(train_scores, perc)
            pred = (test_scores > th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
            sens_rows.append({'Method': name, 'Percentile': perc, 'P': p, 'R': r, 'F1': f1})
    print('\n=== threshold sensitivity (seed 42) ===')
    sens_df = pd.DataFrame(sens_rows)
    pivot = sens_df.pivot(index='Method', columns='Percentile', values='F1').round(4)
    print(pivot)

    out = {
        'seed': SEED,
        'auc_ap': auc_rows,
        'threshold_sensitivity': sens_rows,
    }
    with open('threshold_sensitivity_seed42.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('\nSaved threshold_sensitivity_seed42.json')


if __name__ == '__main__':
    main()
