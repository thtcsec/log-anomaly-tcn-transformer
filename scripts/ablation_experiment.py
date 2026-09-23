"""Ablation studies on HDFS for the VNICT paper.

Covers four axes that reviewers commonly probe:
  (A) Drain3 similarity threshold $\theta$ -> template count and PCA F1.
  (B) Vocabulary growth as a function of parsed log lines.
  (C) DeepLog top-k sensitivity at fixed model.
  (D) Transformer mask ratio sensitivity at fixed architecture.

All experiments use a single fixed seed (42) for speed; multi-seed numbers
already exist in `multi_seed_results.json` / `deeplog_results.json`.
"""

import json
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, flush=True)

MAX_ROWS_BASELINE = 200000
MAX_ROWS_VOCAB = 500000


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


def drain_parse(raw_df, sim_th, depth=4):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = sim_th
    config.drain_depth = depth
    miner = TemplateMiner(config=config)
    template_count_per_rows = []
    parsed = []
    checkpoints = {5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000}
    seen_templates = set()
    for idx, row in enumerate(raw_df.itertuples(index=False), start=1):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
        seen_templates.add(int(result['cluster_id']))
        if idx in checkpoints:
            template_count_per_rows.append((idx, len(seen_templates)))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])
    return events, template_count_per_rows


def block_sequences(events):
    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    return data


def pca_f1(data, seed=SEED):
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=seed, stratify=data['y'])
    vec = CountVectorizer()
    X_train = vec.fit_transform(train_df['text']).toarray()
    X_test = vec.transform(test_df['text']).toarray()
    normal_mask = train_df['y'].values == 0
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_train[normal_mask])
    X_test_scaled = scaler.transform(X_test)
    n_components = max(1, min(20, X_normal.shape[1], X_normal.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_normal)
    train_err = np.mean((X_normal - pca.inverse_transform(pca.transform(X_normal))) ** 2, axis=1)
    threshold = np.percentile(train_err, 95)
    test_err = np.mean((X_test_scaled - pca.inverse_transform(pca.transform(X_test_scaled))) ** 2, axis=1)
    y_pred = (test_err > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_df['y'].values, y_pred, average='binary', zero_division=0)
    return float(p), float(r), float(f1)


def study_drain3(raw_df):
    print('\n=== (A) Drain3 sim_th ablation ===', flush=True)
    rows = []
    for sim_th in [0.3, 0.5, 0.7]:
        events, _ = drain_parse(raw_df, sim_th=sim_th)
        n_templates = events['EventId'].nunique()
        data = block_sequences(events)
        p, r, f1 = pca_f1(data)
        print(f'  sim_th={sim_th}: templates={n_templates}, blocks={len(data)}, PCA F1={f1:.4f}', flush=True)
        rows.append({'sim_th': sim_th, 'templates': int(n_templates), 'blocks': int(len(data)),
                     'precision': p, 'recall': r, 'f1': f1})
    return rows


def study_vocab_growth(raw_df_full):
    print('\n=== (B) Vocabulary growth ===', flush=True)
    _, growth = drain_parse(raw_df_full, sim_th=0.5)
    print('  rows -> templates:')
    for rows, tcount in growth:
        print(f'    {rows:>7d} -> {tcount}', flush=True)
    return [{'rows': int(r), 'templates': int(t)} for r, t in growth]


class WindowDataset(Dataset):
    def __init__(self, sequences, event_to_token, pad_id, window):
        self.windows, self.targets, self.seq_idx = [], [], []
        for s_idx, seq in enumerate(sequences):
            tokens = [event_to_token[e] for e in seq if e in event_to_token]
            if len(tokens) < 2:
                continue
            for i in range(1, len(tokens)):
                start = max(0, i - window)
                ctx = tokens[start:i]
                if len(ctx) < window:
                    ctx = [pad_id] * (window - len(ctx)) + ctx
                self.windows.append(ctx)
                self.targets.append(tokens[i])
                self.seq_idx.append(s_idx)
        self.windows = np.array(self.windows, dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64)
        self.seq_idx = np.array(self.seq_idx, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'seq_idx': self.seq_idx[idx],
        }


class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, pad_id, hidden=64, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=0.1, batch_first=True)
        self.classifier = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.embedding(x)
        out, _ = self.lstm(e)
        return self.classifier(out[:, -1, :])


def study_deeplog_topk(data, ks=(1, 3, 5, 9, 15), window=10, epochs=10):
    print('\n=== (C) DeepLog top-k ablation ===', flush=True)
    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1

    train_df, temp_df = train_test_split(data, test_size=0.4, random_state=SEED, stratify=data['y'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['y'])
    normal_train_df = train_df[train_df['y'] == 0]

    train_ds = WindowDataset(normal_train_df['EventId'].tolist(), event_to_token, pad_id, window)
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id, window)
    y_test = test_df['y'].astype(int).values

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    model = DeepLogLSTM(vocab_size, pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(b['window'].to(DEVICE)), b['target'].to(DEVICE))
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def score_topk(k):
        model.eval()
        n_seq = test_ds.seq_idx.max() + 1
        mismatch = np.zeros(int(n_seq), dtype=np.int64)
        loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        for b in loader:
            logits = model(b['window'].to(DEVICE))
            topk = logits.topk(min(k, vocab_size), dim=-1).indices
            targets = b['target'].to(DEVICE)
            m = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            for s, mm in zip(b['seq_idx'].numpy(), m):
                mismatch[s] += int(mm)
        y_pred = (mismatch > 0).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        return float(p), float(r), float(f1)

    rows = []
    for k in ks:
        p, r, f1 = score_topk(k)
        print(f'  k={k}: P={p:.4f} R={r:.4f} F1={f1:.4f}', flush=True)
        rows.append({'k': int(k), 'precision': p, 'recall': r, 'f1': f1})
    return rows


class LogSequenceDataset(Dataset):
    def __init__(self, df, event_to_token, pad_id, max_len=64):
        self.seqs = df['EventId'].tolist()
        self.labels = df['y'].astype(int).tolist()
        self.event_to_token = event_to_token
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = [self.event_to_token[e] for e in self.seqs[idx] if e in self.event_to_token]
        tokens = tokens[: self.max_len]
        attention = [1] * len(tokens)
        if len(tokens) < self.max_len:
            pad = self.max_len - len(tokens)
            tokens += [self.pad_id] * pad
            attention += [0] * pad
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'attention': torch.tensor(attention, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MiniLogBERT(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, d_model=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, 0.1, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.cls = nn.Linear(d_model, vocab_size)

    def forward(self, x, attn):
        b, l = x.shape
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        h = self.enc(self.tok(x) + self.pos(pos), src_key_padding_mask=(attn == 0))
        return self.cls(h)


def study_transformer_mask(data, mask_ratios=(0.10, 0.15, 0.20, 0.30), max_seq_len=64, epochs=5):
    print('\n=== (D) Transformer mask ratio ablation ===', flush=True)
    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    mask_id = len(event_to_token) + 1
    vocab_size = mask_id + 1

    train_df, temp_df = train_test_split(data, test_size=0.4, random_state=SEED, stratify=data['y'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['y'])
    normal_train_df = train_df[train_df['y'] == 0]

    train_loader = DataLoader(LogSequenceDataset(normal_train_df, event_to_token, pad_id, max_seq_len),
                              batch_size=128, shuffle=True)
    val_loader = DataLoader(LogSequenceDataset(val_df, event_to_token, pad_id, max_seq_len),
                            batch_size=128, shuffle=False)
    test_loader = DataLoader(LogSequenceDataset(test_df, event_to_token, pad_id, max_seq_len),
                             batch_size=128, shuffle=False)

    rows = []
    for mp in mask_ratios:
        torch.manual_seed(SEED)
        model = MiniLogBERT(vocab_size, max_seq_len, pad_id).to(DEVICE)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for _ in range(epochs):
            for b in train_loader:
                tokens = b['tokens'].to(DEVICE)
                attn = b['attention'].to(DEVICE)
                inputs = tokens.clone()
                labels = torch.full_like(tokens, -100)
                rand = torch.rand(tokens.shape, device=DEVICE)
                mask = (rand < mp) & (attn == 1) & (tokens != pad_id)
                for i in range(tokens.size(0)):
                    if attn[i].sum() > 0 and not mask[i].any():
                        valid = torch.where((attn[i] == 1) & (tokens[i] != pad_id))[0]
                        mask[i, valid[0]] = True
                labels[mask] = tokens[mask]
                inputs[mask] = mask_id
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, attn)
                loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        @torch.no_grad()
        def score(loader):
            model.eval()
            scores, labels_y = [], []
            ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            torch.manual_seed(SEED + 999)
            for b in loader:
                tokens = b['tokens'].to(DEVICE)
                attn = b['attention'].to(DEVICE)
                labels_y.extend(b['label'].numpy().tolist())
                rand = torch.rand(tokens.shape, device=DEVICE)
                m = (rand < mp) & (attn == 1) & (tokens != pad_id)
                for i in range(tokens.size(0)):
                    if attn[i].sum() > 0 and not m[i].any():
                        valid = torch.where((attn[i] == 1) & (tokens[i] != pad_id))[0]
                        m[i, valid[0]] = True
                inputs = tokens.clone()
                labels = torch.full_like(tokens, -100)
                labels[m] = tokens[m]
                inputs[m] = mask_id
                logits = model(inputs, attn)
                loss_flat = ce(logits.view(-1, vocab_size), labels.view(-1)).view(tokens.shape)
                denom = m.sum(dim=1).clamp(min=1)
                s = (loss_flat * m.float()).sum(dim=1) / denom
                scores.extend(s.cpu().numpy().tolist())
            return np.array(scores), np.array(labels_y)

        val_scores, val_y = score(val_loader)
        test_scores, test_y = score(test_loader)
        th = np.percentile(val_scores[val_y == 0], 95) if (val_y == 0).any() else 0.0
        y_pred = (test_scores > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(test_y, y_pred, average='binary', zero_division=0)
        print(f'  mask_ratio={mp}: P={p:.4f} R={r:.4f} F1={f1:.4f}', flush=True)
        rows.append({'mask_ratio': float(mp), 'precision': float(p), 'recall': float(r), 'f1': float(f1)})
    return rows


def main():
    t0 = time.perf_counter()
    print(f'Loading {MAX_ROWS_BASELINE} HDFS rows (small)...', flush=True)
    raw_small = stream_rows(MAX_ROWS_BASELINE)
    print(f'Loading {MAX_ROWS_VOCAB} HDFS rows (full for vocab)...', flush=True)
    raw_full = stream_rows(MAX_ROWS_VOCAB)

    drain_rows = study_drain3(raw_small)
    vocab_rows = study_vocab_growth(raw_full)

    events_def, _ = drain_parse(raw_small, sim_th=0.5)
    data_def = block_sequences(events_def)
    deeplog_rows = study_deeplog_topk(data_def)
    transformer_rows = study_transformer_mask(data_def)

    summary = {
        'seed': SEED,
        'drain3_threshold': drain_rows,
        'vocabulary_growth': vocab_rows,
        'deeplog_topk': deeplog_rows,
        'transformer_mask_ratio': transformer_rows,
        'total_seconds': float(time.perf_counter() - t0),
    }
    with open('ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nTotal seconds: {summary["total_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
