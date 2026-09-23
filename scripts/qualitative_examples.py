"""Qualitative anomaly examples for VNICT2026 paper.

Re-runs Drain3 + PCA (seed 42) on HDFS to extract:
  * 3 true anomalous blocks with high PCA score (correctly flagged)
  * 1 true anomalous block with low PCA score (false negative case)
  * 1 normal block with high PCA score (false positive case)

For each block prints:
  block_id, true label, PCA score, percentile rank,
  event ID sequence (E12 E7 ...), and the actual Drain3 template TEXT
  for each event ID (truncated to 80 chars).

Output: qualitative_examples.json + qualitative_examples.tex (LaTeX snippet).
"""
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from datasets import load_dataset
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
MAX_ROWS = 200_000
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4
PCA_COMPONENTS = 20
NORMAL_PERCENTILE = 95

random.seed(SEED)
np.random.seed(SEED)


def stream_rows(max_rows):
    ds = load_dataset('logfit-project/HDFS_v1', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append({
            'content': row['content'],
            'BlockId': row['block_id'],
            'LineAnomaly': int(row['anomaly']),
        })
    return pd.DataFrame(rows)


def drain_parse(df):
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = DRAIN_SIM_TH
    cfg.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=cfg)
    parsed = []
    templates = {}
    for row in df.itertuples():
        r = miner.add_log_message(row.content)
        cid = int(r['cluster_id'])
        if cid not in templates:
            templates[cid] = r['template_mined']
        parsed.append((row.BlockId, int(row.LineAnomaly), cid))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])
    return events, templates, miner


def block_sequences(events):
    g = (events.groupby('BlockId')
         .agg(seq=('EventId', list), anomaly=('LineAnomaly', 'max'))
         .reset_index())
    g['seq_str'] = g['seq'].apply(lambda s: ' '.join(f'E{e}' for e in s))
    return g


print(f'Loading {MAX_ROWS:,} HDFS rows...', flush=True)
df = stream_rows(MAX_ROWS)
print(f'  rows: {len(df):,}  anomaly lines: {df["LineAnomaly"].sum():,}', flush=True)

print('Running Drain3...', flush=True)
events, templates, _ = drain_parse(df)
print(f'  templates: {len(templates)}', flush=True)

data = block_sequences(events)
print(f'  blocks: {len(data):,}  anomalous: {data["anomaly"].sum():,}', flush=True)

vectorizer = CountVectorizer(token_pattern=r'E\d+', lowercase=False)
X = vectorizer.fit_transform(data['seq_str']).toarray()
y = data['anomaly'].to_numpy()

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(data)),
    test_size=0.3, stratify=y, random_state=SEED,
)

normal_mask = (y_tr == 0)
scaler = StandardScaler(with_mean=True)
Xs = scaler.fit_transform(X_tr[normal_mask])
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
pca.fit(Xs)

Xs_te = scaler.transform(X_te)
recon = pca.inverse_transform(pca.transform(Xs_te))
scores = ((Xs_te - recon) ** 2).mean(axis=1)

train_scores = ((Xs - pca.inverse_transform(pca.transform(Xs))) ** 2).mean(axis=1)
tau = np.percentile(train_scores, NORMAL_PERCENTILE)
pred = (scores > tau).astype(int)

print(f'Threshold tau (P{NORMAL_PERCENTILE} of normal train) = {tau:.4f}')
print(f'Test: anomalies={y_te.sum()}, predicted positive={pred.sum()}')

ranks = pd.Series(scores).rank(pct=True).to_numpy()

test_df = pd.DataFrame({
    'orig_idx': idx_te,
    'BlockId': data.iloc[idx_te]['BlockId'].values,
    'seq': data.iloc[idx_te]['seq'].values,
    'true': y_te,
    'score': scores,
    'pred': pred,
    'percentile': ranks,
})


def fmt_template(t, width=80):
    t = ' '.join(t.split())
    return t if len(t) <= width else (t[:width - 1] + '…')


def render_block(row):
    seq = row['seq']
    uniq_in_order = list(dict.fromkeys(seq))
    return {
        'block_id': str(row['BlockId']),
        'true_label': int(row['true']),
        'pred_label': int(row['pred']),
        'pca_score': float(row['score']),
        'percentile': float(row['percentile']),
        'seq_len': len(seq),
        'event_sequence': ' '.join(f'E{e}' for e in seq[:24]) + (' ...' if len(seq) > 24 else ''),
        'unique_events': [f'E{e}' for e in uniq_in_order],
        'templates': {f'E{e}': fmt_template(templates[e]) for e in uniq_in_order},
    }


tp = test_df[(test_df['true'] == 1) & (test_df['pred'] == 1)].sort_values('score', ascending=False)
fn = test_df[(test_df['true'] == 1) & (test_df['pred'] == 0)].sort_values('score', ascending=False)
fp = test_df[(test_df['true'] == 0) & (test_df['pred'] == 1)].sort_values('score', ascending=False)
tn_short = test_df[(test_df['true'] == 0) & (test_df['pred'] == 0)].sort_values('score')

picks = {
    'true_positive_top1': tp.iloc[0] if len(tp) else None,
    'true_positive_top2': tp.iloc[len(tp) // 2] if len(tp) > 1 else None,
    'true_positive_top3': tp.iloc[-1] if len(tp) > 2 else None,
    'false_negative_top1': fn.iloc[0] if len(fn) else None,
    'false_positive_top1': fp.iloc[0] if len(fp) else None,
}

out = {
    'protocol': {
        'seed': SEED, 'max_rows': MAX_ROWS,
        'drain_sim_th': DRAIN_SIM_TH, 'drain_depth': DRAIN_DEPTH,
        'pca_components': PCA_COMPONENTS,
        'threshold_percentile': NORMAL_PERCENTILE,
        'tau_pca': float(tau),
        'n_templates': len(templates),
        'test_anomalies': int(y_te.sum()),
        'test_positives_pred': int(pred.sum()),
    },
    'examples': {}
}

for label, row in picks.items():
    if row is None:
        continue
    rendered = render_block(row)
    out['examples'][label] = rendered
    print(f'\n--- {label} ---')
    print(f'  Block: {rendered["block_id"]}')
    print(f'  true={rendered["true_label"]}  pred={rendered["pred_label"]}'
          f'  score={rendered["pca_score"]:.4f}'
          f'  pct={rendered["percentile"]:.3f}  len={rendered["seq_len"]}')
    print(f'  seq: {rendered["event_sequence"]}')
    for eid, tmpl in rendered['templates'].items():
        print(f'    {eid}: {tmpl}')

with open('qualitative_examples.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\nSaved -> qualitative_examples.json')
