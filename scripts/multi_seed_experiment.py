"""Multi-seed experiment for VNICT2026 paper.

Loads HDFS_v1 once, runs Drain3 parsing once, then evaluates PCA, TruncatedSVD,
Isolation Forest, and a LogBERT-inspired Transformer across multiple seeds.
Outputs `multi_seed_results.json` and `multi_seed_results.csv` with mean/std.
"""

import json
import time
import os
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEEDS = [21, 42, 84, 123, 777]
MAX_ROWS_BASELINE = 500000
MAX_ROWS_TRANSFORMER = 200000
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4

MAX_SEQ_LEN = 64
MASK_PROB = 0.15
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
FF_DIM = 128
SCORE_PASSES = 5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, flush=True)


def stream_rows(max_rows):
    ds = load_dataset('logfit-project/HDFS_v1', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append({
            'content': row['content'],
            'BlockId': row['block_id'],
            'LineAnomaly': int(row['anomaly'])
        })
    return pd.DataFrame(rows)


def drain_parse(raw_df, label='parse'):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=config)
    parsed = []
    start = time.perf_counter()
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc=label):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])
    elapsed = time.perf_counter() - start
    print(f'{label}: {len(events)} rows, {events["EventId"].nunique()} templates, {elapsed:.2f}s', flush=True)
    return events, elapsed


def block_sequences(events):
    data = (
        events.groupby('BlockId')
        .agg(
            EventId=('EventId', list),
            SeqLen=('EventId', 'size'),
            y=('LineAnomaly', 'max'),
        )
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    return data


def run_baselines(data, seed):
    """Run PCA, TruncatedSVD, Isolation Forest with a given seed.

    Returns a list of dicts with method/seed/precision/recall/f1.
    """
    np.random.seed(seed)
    train_df, test_df = train_test_split(
        data, test_size=0.3, random_state=seed, stratify=data['y']
    )
    y_test = test_df['y'].values
    out = []

    # PCA on dense bag-of-events.
    vec_pca = CountVectorizer()
    X_train_all = vec_pca.fit_transform(train_df['text']).toarray()
    X_test_all = vec_pca.transform(test_df['text']).toarray()
    normal_mask = train_df['y'].values == 0

    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_train_all[normal_mask])
    X_test_scaled = scaler.transform(X_test_all)
    n_components = max(1, min(20, X_normal.shape[1], X_normal.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_normal)
    X_train_recon = pca.inverse_transform(pca.transform(X_normal))
    train_err = np.mean((X_normal - X_train_recon) ** 2, axis=1)
    threshold = np.percentile(train_err, 95)
    X_test_recon = pca.inverse_transform(pca.transform(X_test_scaled))
    test_err = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)
    y_pred = (test_err > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    out.append({'Method': 'PCA', 'Seed': seed, 'Precision': float(p), 'Recall': float(r), 'F1': float(f1)})

    # TruncatedSVD on sparse bag-of-events.
    vec_svd = CountVectorizer()
    X_train_sp = vec_svd.fit_transform(train_df['text'])
    X_test_sp = vec_svd.transform(test_df['text'])
    X_normal_sp = X_train_sp[normal_mask]
    n_components_svd = max(1, min(20, X_normal_sp.shape[1] - 1, X_normal_sp.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components_svd, random_state=seed)
    svd.fit(X_normal_sp)
    def svd_err(X):
        Z = svd.transform(X)
        Xh = svd.inverse_transform(Z)
        Xd = X.toarray()
        return np.mean((Xd - Xh) ** 2, axis=1)
    train_svd_err = svd_err(X_normal_sp)
    test_svd_err = svd_err(X_test_sp)
    svd_th = np.percentile(train_svd_err, 95)
    y_pred = (test_svd_err > svd_th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    out.append({'Method': 'TruncatedSVD', 'Seed': seed, 'Precision': float(p), 'Recall': float(r), 'F1': float(f1)})

    # Isolation Forest on TF-IDF.
    vec_if = TfidfVectorizer()
    normal_train = train_df[train_df['y'] == 0]
    X_if_train = vec_if.fit_transform(normal_train['text'])
    X_if_test = vec_if.transform(test_df['text'])
    contamination = max(0.001, min(0.2, train_df['y'].mean()))
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1
    )
    iso.fit(X_if_train)
    pred = iso.predict(X_if_test)
    y_pred = (pred == -1).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    out.append({'Method': 'Isolation Forest', 'Seed': seed, 'Precision': float(p), 'Recall': float(r), 'F1': float(f1)})

    return out


def encode_sequence(event_ids, event_to_token, pad_id, max_len=MAX_SEQ_LEN):
    tokens = [event_to_token[eid] for eid in event_ids if eid in event_to_token]
    tokens = tokens[:max_len]
    attention = [1] * len(tokens)
    if len(tokens) < max_len:
        pad_len = max_len - len(tokens)
        tokens += [pad_id] * pad_len
        attention += [0] * pad_len
    return np.array(tokens, dtype=np.int64), np.array(attention, dtype=np.int64)


class LogSequenceDataset(Dataset):
    def __init__(self, df, event_to_token, pad_id):
        self.seqs = df['EventId'].tolist()
        self.labels = df['y'].astype(int).tolist()
        self.event_to_token = event_to_token
        self.pad_id = pad_id

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens, attention = encode_sequence(self.seqs[idx], self.event_to_token, self.pad_id)
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'attention': torch.tensor(attention, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MiniLogBERT(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, d_model=64, nhead=4, num_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        key_padding_mask = attention == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.classifier(h)


def make_masked_batch(tokens, attention, mask_id, pad_id, mask_prob=MASK_PROB):
    inputs = tokens.clone()
    labels = torch.full_like(tokens, fill_value=-100)
    rand = torch.rand(tokens.shape, device=tokens.device)
    mask = (rand < mask_prob) & (attention == 1) & (tokens != pad_id)
    for i in range(tokens.size(0)):
        if attention[i].sum() > 0 and not mask[i].any():
            valid_positions = torch.where((attention[i] == 1) & (tokens[i] != pad_id))[0]
            chosen_idx = torch.randint(len(valid_positions), (1,), device=tokens.device).item()
            mask[i, valid_positions[chosen_idx]] = True
    labels[mask] = tokens[mask]
    inputs[mask] = mask_id
    return inputs, labels


def run_transformer(data, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_event_ids = sorted(set(e for seq in data['EventId'] for e in seq))
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    mask_id = len(event_to_token) + 1
    vocab_size = mask_id + 1

    train_df, temp_df = train_test_split(
        data, test_size=0.4, random_state=seed, stratify=data['y']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df['y']
    )
    normal_train_df = train_df[train_df['y'] == 0].copy()

    train_loader = DataLoader(LogSequenceDataset(normal_train_df, event_to_token, pad_id),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(LogSequenceDataset(val_df, event_to_token, pad_id),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(LogSequenceDataset(test_df, event_to_token, pad_id),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MiniLogBERT(
        vocab_size=vocab_size, max_len=MAX_SEQ_LEN, pad_id=pad_id,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, ff_dim=FF_DIM,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Seed {seed}: vocab={vocab_size}, params={num_params}', flush=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in train_loader:
            tokens = batch['tokens'].to(DEVICE)
            attention = batch['attention'].to(DEVICE)
            inputs, labels = make_masked_batch(tokens, attention, mask_id, pad_id)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, attention)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f'  seed {seed} epoch {epoch}: loss={np.mean(losses):.4f}', flush=True)

    @torch.no_grad()
    def score_once(loader, seed_offset):
        model.eval()
        all_scores = []
        all_labels = []
        ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        torch.manual_seed(seed + seed_offset)
        for batch in loader:
            tokens = batch['tokens'].to(DEVICE)
            attention = batch['attention'].to(DEVICE)
            labels_y = batch['label'].cpu().numpy().tolist()
            inputs = tokens.clone()
            labels = torch.full_like(tokens, fill_value=-100)
            rand = torch.rand(tokens.shape, device=DEVICE)
            mask = (rand < MASK_PROB) & (attention == 1) & (tokens != pad_id)
            for i in range(tokens.size(0)):
                if attention[i].sum() > 0 and not mask[i].any():
                    valid_positions = torch.where((attention[i] == 1) & (tokens[i] != pad_id))[0]
                    mask[i, valid_positions[0]] = True
            labels[mask] = tokens[mask]
            inputs[mask] = mask_id
            logits = model(inputs, attention)
            loss_flat = ce_none(logits.view(-1, vocab_size), labels.view(-1)).view(tokens.size(0), tokens.size(1))
            denom = mask.sum(dim=1).clamp(min=1)
            scores = (loss_flat * mask.float()).sum(dim=1) / denom
            all_scores.extend(scores.detach().cpu().numpy().tolist())
            all_labels.extend(labels_y)
        return np.array(all_scores), np.array(all_labels)

    @torch.no_grad()
    def score_avg(loader):
        score_list = []
        labels_ref = None
        for pidx in range(SCORE_PASSES):
            scores, labels = score_once(loader, seed_offset=999 + pidx)
            score_list.append(scores)
            labels_ref = labels
        return np.mean(np.vstack(score_list), axis=0), labels_ref

    val_scores, val_y = score_avg(val_loader)
    test_scores, test_y = score_avg(test_loader)

    normal_val_scores = val_scores[val_y == 0]
    threshold_unsup = np.percentile(normal_val_scores, 95)
    pred_unsup = (test_scores > threshold_unsup).astype(int)
    p_u, r_u, f1_u, _ = precision_recall_fscore_support(test_y, pred_unsup, average='binary', zero_division=0)

    candidate_percentiles = np.linspace(50, 99, 50)
    best = {'percentile': None, 'threshold': None, 'f1': -1.0}
    for perc in candidate_percentiles:
        th = np.percentile(val_scores, perc)
        val_pred = (val_scores > th).astype(int)
        _, _, f1_v, _ = precision_recall_fscore_support(val_y, val_pred, average='binary', zero_division=0)
        if f1_v > best['f1']:
            best = {'percentile': float(perc), 'threshold': float(th), 'f1': float(f1_v)}
    pred_tuned = (test_scores > best['threshold']).astype(int)
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(test_y, pred_tuned, average='binary', zero_division=0)

    return [
        {'Method': 'Transformer (unsup. threshold)', 'Seed': seed,
         'Precision': float(p_u), 'Recall': float(r_u), 'F1': float(f1_u),
         'num_params': int(num_params)},
        {'Method': 'Transformer (val-tuned threshold)', 'Seed': seed,
         'Precision': float(p_t), 'Recall': float(r_t), 'F1': float(f1_t),
         'num_params': int(num_params)},
    ]


def main():
    t0 = time.perf_counter()
    print('=== loading 500k rows for baseline ===', flush=True)
    raw_baseline = stream_rows(MAX_ROWS_BASELINE)
    print('loaded', len(raw_baseline), 'rows', flush=True)
    events_baseline, _ = drain_parse(raw_baseline, label='drain3-500k')
    data_baseline = block_sequences(events_baseline)
    print(f'baseline blocks: {len(data_baseline)}', flush=True)

    print('=== loading 200k rows for transformer ===', flush=True)
    raw_tx = raw_baseline.iloc[:MAX_ROWS_TRANSFORMER].copy()
    events_tx, _ = drain_parse(raw_tx, label='drain3-200k')
    data_tx = block_sequences(events_tx)
    print(f'transformer blocks: {len(data_tx)}', flush=True)

    all_rows = []
    for seed in SEEDS:
        print(f'\n--- baselines seed {seed} ---', flush=True)
        all_rows.extend(run_baselines(data_baseline, seed))
        for r in all_rows[-3:]:
            print(f"  {r['Method']:<18} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)

    for seed in SEEDS:
        print(f'\n--- transformer seed {seed} ---', flush=True)
        all_rows.extend(run_transformer(data_tx, seed))
        for r in all_rows[-2:]:
            print(f"  {r['Method']:<40} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv('multi_seed_results.csv', index=False)

    agg = (
        df.groupby('Method')[['Precision', 'Recall', 'F1']]
        .agg(['mean', 'std'])
        .round(4)
    )
    print('\n=== aggregated ===', flush=True)
    print(agg, flush=True)

    summary = {
        'seeds': SEEDS,
        'max_rows_baseline': MAX_ROWS_BASELINE,
        'max_rows_transformer': MAX_ROWS_TRANSFORMER,
        'baseline_blocks': int(len(data_baseline)),
        'baseline_normal': int((data_baseline['y'] == 0).sum()),
        'baseline_anomaly': int((data_baseline['y'] == 1).sum()),
        'baseline_event_templates': int(events_baseline['EventId'].nunique()),
        'transformer_blocks': int(len(data_tx)),
        'transformer_event_templates': int(events_tx['EventId'].nunique()),
        'rows': all_rows,
        'aggregated': {
            method: {
                'Precision_mean': float(df[df.Method == method]['Precision'].mean()),
                'Precision_std': float(df[df.Method == method]['Precision'].std(ddof=1)),
                'Recall_mean': float(df[df.Method == method]['Recall'].mean()),
                'Recall_std': float(df[df.Method == method]['Recall'].std(ddof=1)),
                'F1_mean': float(df[df.Method == method]['F1'].mean()),
                'F1_std': float(df[df.Method == method]['F1'].std(ddof=1)),
            }
            for method in df['Method'].unique()
        },
        'total_seconds': float(time.perf_counter() - t0),
    }
    with open('multi_seed_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('\nSaved multi_seed_results.csv and multi_seed_results.json', flush=True)
    print(f'Total seconds: {summary["total_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
