"""DeepLog (LSTM next-event prediction) baseline on HDFS.

Implements the standard DeepLog formulation from Du et al. (CCS 2017):
- Sliding window of g previous events predicts the next event.
- LSTM (2 layers, hidden 64) outputs softmax over event vocabulary.
- Trained on normal sequences only with cross-entropy.
- At inference, a position is anomalous if the actual next event is not
  in the top-k predicted candidates; a sequence is anomalous if it
  contains at least one anomalous position.
- We additionally report a continuous score (mean negative log-likelihood)
  thresholded at the 95th percentile of normal-validation scores so the
  comparison with PCA / TruncatedSVD / Transformer is metric-consistent.
"""

import json
import time
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

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEEDS = [21, 42, 84, 123, 777]
MAX_ROWS = 200000
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4

WINDOW = 10
TOP_K = 9
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

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
            'LineAnomaly': int(row['anomaly']),
        })
    return pd.DataFrame(rows)


def drain_parse(raw_df):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=config)
    parsed = []
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc='drain3'):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    return pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])


def block_sequences(events):
    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    return data


class WindowDataset(Dataset):
    """Materialises (window, next_event) pairs for LSTM training/scoring."""

    def __init__(self, sequences, event_to_token, pad_id, window=WINDOW):
        self.window = window
        self.pad_id = pad_id
        self.event_to_token = event_to_token
        self.windows = []
        self.targets = []
        self.seq_idx = []
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
    def __init__(self, vocab_size, pad_id, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, windows):
        x = self.embedding(windows)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, vocab_size, pad_id, embed_dim=64, num_channels=[64, 64, 64], kernel_size=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.tcn = TemporalConvNet(embed_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.classifier = nn.Linear(num_channels[-1], vocab_size)

    def forward(self, windows):
        x = self.embedding(windows)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        last_step = y[:, :, -1]
        return self.classifier(last_step)



def run_deeplog(data, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1

    train_df, temp_df = train_test_split(data, test_size=0.4, random_state=seed, stratify=data['y'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['y'])
    normal_train_df = train_df[train_df['y'] == 0]

    train_ds = WindowDataset(normal_train_df['EventId'].tolist(), event_to_token, pad_id)
    val_ds = WindowDataset(val_df['EventId'].tolist(), event_to_token, pad_id)
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id)

    val_labels = val_df['y'].astype(int).values
    test_labels = test_df['y'].astype(int).values

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = DeepLogLSTM(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Seed {seed}: vocab={vocab_size}, params={num_params}, train_windows={len(train_ds)}', flush=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in train_loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == EPOCHS or epoch % 3 == 0:
            print(f'  seed {seed} epoch {epoch}: loss={np.mean(losses):.4f}', flush=True)

    @torch.no_grad()
    def score_dataset(ds):
        model.eval()
        n_seq = ds.seq_idx.max() + 1 if len(ds) > 0 else 0
        seq_mismatch_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_window_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_nll_sum = np.zeros(int(n_seq), dtype=np.float64)

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            seq_idx = batch['seq_idx'].numpy()
            logits = model(windows)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = logits.topk(TOP_K, dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            for s, m, lp in zip(seq_idx, mismatch, target_log_probs):
                seq_mismatch_count[s] += int(m)
                seq_window_count[s] += 1
                seq_nll_sum[s] += float(-lp)

        seq_window_count = np.maximum(seq_window_count, 1)
        binary_score = (seq_mismatch_count > 0).astype(int)
        ratio_score = seq_mismatch_count / seq_window_count
        nll_score = seq_nll_sum / seq_window_count
        return binary_score, ratio_score, nll_score

    val_binary, val_ratio, val_nll = score_dataset(val_ds)
    test_binary, test_ratio, test_nll = score_dataset(test_ds)

    p_b, r_b, f1_b, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)

    val_nll_normal = val_nll[val_labels == 0]
    nll_threshold = np.percentile(val_nll_normal, 95) if len(val_nll_normal) > 0 else 0.0
    test_nll_pred = (test_nll > nll_threshold).astype(int)
    p_n, r_n, f1_n, _ = precision_recall_fscore_support(test_labels, test_nll_pred, average='binary', zero_division=0)

    return [
        {
            'Method': 'DeepLog (top-k binary)',
            'Seed': seed,
            'Precision': float(p_b), 'Recall': float(r_b), 'F1': float(f1_b),
            'top_k': TOP_K, 'window': WINDOW, 'num_params': int(num_params),
        },
        {
            'Method': 'DeepLog (NLL threshold)',
            'Seed': seed,
            'Precision': float(p_n), 'Recall': float(r_n), 'F1': float(f1_n),
            'top_k': TOP_K, 'window': WINDOW, 'num_params': int(num_params),
            'nll_threshold': float(nll_threshold),
        },
    ]


def run_tcn(data, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1

    train_df, temp_df = train_test_split(data, test_size=0.4, random_state=seed, stratify=data['y'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['y'])
    normal_train_df = train_df[train_df['y'] == 0]

    train_ds = WindowDataset(normal_train_df['EventId'].tolist(), event_to_token, pad_id)
    val_ds = WindowDataset(val_df['EventId'].tolist(), event_to_token, pad_id)
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id)

    val_labels = val_df['y'].astype(int).values
    test_labels = test_df['y'].astype(int).values

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TCNModel(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'TCN Seed {seed}: vocab={vocab_size}, params={num_params}, train_windows={len(train_ds)}', flush=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in train_loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == EPOCHS or epoch % 3 == 0:
            print(f'  TCN seed {seed} epoch {epoch}: loss={np.mean(losses):.4f}', flush=True)

    @torch.no_grad()
    def score_dataset(ds):
        model.eval()
        n_seq = ds.seq_idx.max() + 1 if len(ds) > 0 else 0
        seq_mismatch_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_window_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_nll_sum = np.zeros(int(n_seq), dtype=np.float64)

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            seq_idx = batch['seq_idx'].numpy()
            logits = model(windows)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = logits.topk(min(TOP_K, vocab_size), dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            for s, m, lp in zip(seq_idx, mismatch, target_log_probs):
                seq_mismatch_count[s] += int(m)
                seq_window_count[s] += 1
                seq_nll_sum[s] += float(-lp)

        seq_window_count = np.maximum(seq_window_count, 1)
        binary_score = (seq_mismatch_count > 0).astype(int)
        ratio_score = seq_mismatch_count / seq_window_count
        nll_score = seq_nll_sum / seq_window_count
        return binary_score, ratio_score, nll_score

    val_binary, val_ratio, val_nll = score_dataset(val_ds)
    test_binary, test_ratio, test_nll = score_dataset(test_ds)

    p_b, r_b, f1_b, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)

    val_nll_normal = val_nll[val_labels == 0]
    nll_threshold = np.percentile(val_nll_normal, 95) if len(val_nll_normal) > 0 else 0.0
    test_nll_pred = (test_nll > nll_threshold).astype(int)
    p_n, r_n, f1_n, _ = precision_recall_fscore_support(test_labels, test_nll_pred, average='binary', zero_division=0)

    return [
        {
            'Method': 'TCN (top-k binary)',
            'Seed': seed,
            'Precision': float(p_b), 'Recall': float(r_b), 'F1': float(f1_b),
            'top_k': TOP_K, 'window': WINDOW, 'num_params': int(num_params),
        },
        {
            'Method': 'TCN (NLL threshold)',
            'Seed': seed,
            'Precision': float(p_n), 'Recall': float(r_n), 'F1': float(f1_n),
            'top_k': TOP_K, 'window': WINDOW, 'num_params': int(num_params),
            'nll_threshold': float(nll_threshold),
        },
    ]


def main():
    t0 = time.perf_counter()
    print(f'Loading {MAX_ROWS} HDFS rows...', flush=True)
    raw_df = stream_rows(MAX_ROWS)
    print('rows:', len(raw_df), flush=True)

    events = drain_parse(raw_df)
    print(f'events: {len(events)}, templates: {events["EventId"].nunique()}', flush=True)
    data = block_sequences(events)
    print(f'blocks: {len(data)} (normal={(data.y == 0).sum()}, anomaly={(data.y == 1).sum()})', flush=True)
    print(f'sequence length: mean={data["SeqLen"].mean():.2f}, max={data["SeqLen"].max()}', flush=True)

    all_rows = []
    for seed in SEEDS:
        print(f'\n=== seed {seed} ===', flush=True)
        rows_dl = run_deeplog(data, seed)
        for r in rows_dl:
            print(f"  {r['Method']:<28} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows_dl)

        rows_tcn = run_tcn(data, seed)
        for r in rows_tcn:
            print(f"  {r['Method']:<28} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows_tcn)


    df = pd.DataFrame(all_rows)
    df.to_csv('deeplog_results.csv', index=False)

    summary = {
        'config': {
            'seeds': SEEDS,
            'max_rows': MAX_ROWS,
            'window': WINDOW,
            'top_k': TOP_K,
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
        },
        'dataset': {
            'blocks': int(len(data)),
            'normal': int((data['y'] == 0).sum()),
            'anomaly': int((data['y'] == 1).sum()),
            'event_templates': int(events['EventId'].nunique()),
        },
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
    with open('deeplog_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n=== aggregated ===', flush=True)
    agg = df.groupby('Method')[['Precision', 'Recall', 'F1']].agg(['mean', 'std']).round(4)
    print(agg, flush=True)
    print(f'\nTotal seconds: {summary["total_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
