"""BGL grouping strategy benchmark.
Evaluates PCA, DeepLog, and TCN on BGL with different window sizes (50, 100, 200 non-overlap)
and sliding window (W=100, stride=50) across 5 seeds.
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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEEDS = [21, 42, 84, 123, 777]
MAX_ROWS = 200000
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4

DL_WINDOW = 10
DL_TOP_K = 9
DL_BATCH = 512
DL_EPOCHS = 5
DL_LR = 1e-3
D_MODEL = 64
N_LAYERS = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, flush=True)

def stream_rows(max_rows):
    ds = load_dataset('logfit-project/BGL', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append({
            'content': row.get('content', row.get('Content', '')),
            'anomaly': int(row.get('anomaly', 0)),
        })
    return pd.DataFrame(rows)

def drain_parse(raw_df):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=config)
    parsed = []
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc='drain3-bgl'):
        result = miner.add_log_message(row.content)
        parsed.append((int(row.anomaly), int(result['cluster_id'])))
    return pd.DataFrame(parsed, columns=['LineAnomaly', 'EventId'])

def group_sequences(events, window, stride):
    rows = []
    total = len(events)
    for start in range(0, total - window + 1, stride):
        chunk = events.iloc[start:start + window]
        rows.append({
            'GroupKey': f'W{start}',
            'EventId': chunk['EventId'].tolist(),
            'SeqLen': int(window),
            'y': int(chunk['LineAnomaly'].max()),
        })
    out = pd.DataFrame(rows)
    out['text'] = out['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    return out

# --- Models & Datasets ---

class WindowDataset(Dataset):
    def __init__(self, sequences, event_to_token, pad_id, window=DL_WINDOW):
        self.window = window
        self.pad_id = pad_id
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
        self.windows = np.array(self.windows, dtype=np.int64) if self.windows else np.zeros((0, window), dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64) if self.targets else np.zeros((0,), dtype=np.int64)
        self.seq_idx = np.array(self.seq_idx, dtype=np.int64) if self.seq_idx else np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'seq_idx': self.seq_idx[idx],
        }

class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=pad_id)
        self.lstm = nn.LSTM(D_MODEL, D_MODEL, num_layers=N_LAYERS, dropout=0.1, batch_first=True)
        self.classifier = nn.Linear(D_MODEL, vocab_size)

    def forward(self, windows):
        x = self.embedding(windows)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

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
    def __init__(self, vocab_size, pad_id, embed_dim=D_MODEL, num_channels=[D_MODEL, D_MODEL, D_MODEL], kernel_size=2, dropout=0.1):
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

# --- Experiment Runners ---

def run_pca(data, seed):
    np.random.seed(seed)
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=seed, stratify=data['y'])
    y_test = test_df['y'].values

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

    X_train_recon = pca.inverse_transform(pca.transform(X_normal))
    train_err = np.mean((X_normal - X_train_recon) ** 2, axis=1)
    threshold = np.percentile(train_err, 95)

    X_test_recon = pca.inverse_transform(pca.transform(X_test_scaled))
    test_err = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)
    y_pred = (test_err > threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    return float(f1)

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
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id)
    test_labels = test_df['y'].astype(int).values

    train_loader = DataLoader(train_ds, batch_size=DL_BATCH, shuffle=True, num_workers=0)
    model = DeepLogLSTM(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, DL_EPOCHS + 1):
        for batch in train_loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    @torch.no_grad()
    def score(ds):
        model.eval()
        n_seq = ds.seq_idx.max() + 1 if len(ds) > 0 else 0
        seq_mismatch = np.zeros(int(n_seq), dtype=np.int64)
        loader = DataLoader(ds, batch_size=DL_BATCH, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            sidx = batch['seq_idx'].numpy()
            logits = model(windows)
            topk = logits.topk(min(DL_TOP_K, vocab_size), dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            for s, m in zip(sidx, mismatch):
                seq_mismatch[s] += int(m)
        return (seq_mismatch > 0).astype(int)

    test_binary = score(test_ds)
    _, _, f1, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)
    return float(f1)

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
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id)
    test_labels = test_df['y'].astype(int).values

    train_loader = DataLoader(train_ds, batch_size=DL_BATCH, shuffle=True, num_workers=0)
    model = TCNModel(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, DL_EPOCHS + 1):
        for batch in train_loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    @torch.no_grad()
    def score(ds):
        model.eval()
        n_seq = ds.seq_idx.max() + 1 if len(ds) > 0 else 0
        seq_mismatch = np.zeros(int(n_seq), dtype=np.int64)
        loader = DataLoader(ds, batch_size=DL_BATCH, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            sidx = batch['seq_idx'].numpy()
            logits = model(windows)
            topk = logits.topk(min(DL_TOP_K, vocab_size), dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            for s, m in zip(sidx, mismatch):
                seq_mismatch[s] += int(m)
        return (seq_mismatch > 0).astype(int)

    test_binary = score(test_ds)
    _, _, f1, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)
    return float(f1)

# --- Main ---

def main():
    print(f'Loading {MAX_ROWS} BGL rows...', flush=True)
    raw_df = stream_rows(MAX_ROWS)
    print('Parsing logs with Drain3...', flush=True)
    events = drain_parse(raw_df)
    
    configs = [
        {'name': 'W=50', 'window': 50, 'stride': 50},
        {'name': 'W=100', 'window': 100, 'stride': 100},
        {'name': 'W=200', 'window': 200, 'stride': 200},
        {'name': 'W=100, stride=50', 'window': 100, 'stride': 50},
    ]
    
    results = {}
    
    for cfg in configs:
        name = cfg['name']
        print(f'\nRunning configuration: {name}', flush=True)
        data = group_sequences(events, cfg['window'], cfg['stride'])
        print(f'  Total sequences: {len(data)} (anomaly rate: {data.y.mean():.4f})', flush=True)
        
        results[name] = {
            'PCA': [],
            'DeepLog': [],
            'TCN': []
        }
        
        for seed in SEEDS:
            print(f'  Seed {seed}...', flush=True)
            
            # PCA
            f1_pca = run_pca(data, seed)
            results[name]['PCA'].append(f1_pca)
            
            # DeepLog
            f1_dl = run_deeplog(data, seed)
            results[name]['DeepLog'].append(f1_dl)
            
            # TCN
            f1_tcn = run_tcn(data, seed)
            results[name]['TCN'].append(f1_tcn)
            
            print(f'    PCA F1: {f1_pca:.4f} | DeepLog F1: {f1_dl:.4f} | TCN F1: {f1_tcn:.4f}', flush=True)

    # Compile Table Results
    print('\n=== Final Grouping Benchmark Results (F1-score) ===', flush=True)
    table_rows = []
    for name in results:
        pca_mean = np.mean(results[name]['PCA'])
        pca_std = np.std(results[name]['PCA'], ddof=1) if len(results[name]['PCA']) > 1 else 0.0
        dl_mean = np.mean(results[name]['DeepLog'])
        dl_std = np.std(results[name]['DeepLog'], ddof=1) if len(results[name]['DeepLog']) > 1 else 0.0
        tcn_mean = np.mean(results[name]['TCN'])
        tcn_std = np.std(results[name]['TCN'], ddof=1) if len(results[name]['TCN']) > 1 else 0.0
        
        row_str = f"{name:<20} | PCA: {pca_mean:.4f} ± {pca_std:.4f} | DeepLog: {dl_mean:.4f} ± {dl_std:.4f} | TCN: {tcn_mean:.4f} ± {tcn_std:.4f}"
        print(row_str, flush=True)
        
        table_rows.append({
            'Configuration': name,
            'PCA': f"{pca_mean:.4f} ± {pca_std:.4f}",
            'DeepLog': f"{dl_mean:.4f} ± {dl_std:.4f}",
            'TCN': f"{tcn_mean:.4f} ± {tcn_std:.4f}"
        })
        
    with open('bgl_grouping_results.json', 'w', encoding='utf-8') as f:
        json.dump(table_rows, f, indent=2, ensure_ascii=False)
    print('\nSaved to bgl_grouping_results.json', flush=True)

if __name__ == '__main__':
    main()
