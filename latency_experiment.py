"""Inference latency benchmark on HDFS for VNICT paper.

Measures per-sample inference time on a held-out test set for:
- PCA, TruncatedSVD, Isolation Forest (CPU)
- DeepLog LSTM (top-k scoring) on CPU and GPU
- Transformer masked event modeling on CPU and GPU

Reports mean latency in milliseconds and effective throughput (samples/sec)
for batch size 1 to simulate streaming inference, and batch 128 for bulk.
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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE_GPU = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_CPU = 'cpu'

MAX_ROWS = 200000
N_BENCH_SAMPLES = 1000
WARMUP = 50

print('GPU device:', DEVICE_GPU, flush=True)


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
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    miner = TemplateMiner(config=config)
    parsed = []
    for row in raw_df.itertuples(index=False):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    return pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId']), miner


def block_sequences(events):
    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    return data


def bench(callable_, n_samples=N_BENCH_SAMPLES, warmup=WARMUP):
    for _ in range(warmup):
        callable_()
    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        callable_()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {'mean_ms': float(arr.mean()), 'std_ms': float(arr.std()), 'median_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)), 'p99_ms': float(np.percentile(arr, 99)),
            'throughput_per_sec': float(1000.0 / arr.mean())}


def main():
    t0 = time.perf_counter()
    print(f'Loading {MAX_ROWS} HDFS rows...', flush=True)
    raw = stream_rows(MAX_ROWS)
    events, miner = drain_parse(raw)
    data = block_sequences(events)
    print(f'blocks={len(data)} templates={events["EventId"].nunique()}', flush=True)

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=SEED, stratify=data['y'])
    test_sample = test_df.sample(n=min(N_BENCH_SAMPLES, len(test_df)), random_state=SEED).reset_index(drop=True)

    vec = CountVectorizer()
    X_train = vec.fit_transform(train_df['text']).toarray()
    X_test_full = vec.transform(test_sample['text']).toarray()
    normal_mask = train_df['y'].values == 0
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_train[normal_mask])
    X_test_scaled = scaler.transform(X_test_full)
    pca = PCA(n_components=min(20, X_normal.shape[1], X_normal.shape[0] - 1), random_state=SEED)
    pca.fit(X_normal)

    counter = {'i': 0}

    def pca_step():
        x = X_test_scaled[counter['i'] % len(X_test_scaled)].reshape(1, -1)
        z = pca.transform(x)
        xh = pca.inverse_transform(z)
        _ = float(np.mean((x - xh) ** 2))
        counter['i'] += 1

    pca_lat = bench(pca_step)
    print(f'PCA: {pca_lat}', flush=True)

    vec_svd = CountVectorizer()
    X_train_sp = vec_svd.fit_transform(train_df['text'])
    X_test_sp = vec_svd.transform(test_sample['text'])
    X_normal_sp = X_train_sp[normal_mask]
    svd = TruncatedSVD(n_components=min(20, X_normal_sp.shape[1] - 1, X_normal_sp.shape[0] - 1), random_state=SEED)
    svd.fit(X_normal_sp)

    counter['i'] = 0

    def svd_step():
        x = X_test_sp[counter['i'] % X_test_sp.shape[0]]
        z = svd.transform(x)
        xh = svd.inverse_transform(z)
        _ = float(np.mean((x.toarray() - xh) ** 2))
        counter['i'] += 1

    svd_lat = bench(svd_step)
    print(f'SVD: {svd_lat}', flush=True)

    vec_if = TfidfVectorizer()
    X_if_train = vec_if.fit_transform(train_df[train_df['y'] == 0]['text'])
    X_if_test = vec_if.transform(test_sample['text'])
    iso = IsolationForest(n_estimators=200, contamination=max(0.001, min(0.2, train_df['y'].mean())),
                          random_state=SEED, n_jobs=1)
    iso.fit(X_if_train)

    counter['i'] = 0

    def if_step():
        x = X_if_test[counter['i'] % X_if_test.shape[0]]
        _ = iso.predict(x)[0]
        counter['i'] += 1

    if_lat = bench(if_step)
    print(f'IsolationForest: {if_lat}', flush=True)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1
    WINDOW = 10

    class DeepLogLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 64, padding_idx=pad_id)
            self.lstm = nn.LSTM(64, 64, num_layers=2, dropout=0.1, batch_first=True)
            self.classifier = nn.Linear(64, vocab_size)

        def forward(self, x):
            e = self.embedding(x)
            o, _ = self.lstm(e)
            return self.classifier(o[:, -1, :])

    deeplog_results = {}
    for device in [DEVICE_CPU, DEVICE_GPU] if DEVICE_GPU == 'cuda' else [DEVICE_CPU]:
        model = DeepLogLSTM().to(device).eval()
        normal_train = train_df[train_df['y'] == 0]
        seqs = normal_train['EventId'].tolist()[:5000]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        windows_train, targets_train = [], []
        for seq in seqs:
            tokens = [event_to_token[e] for e in seq if e in event_to_token]
            if len(tokens) < 2:
                continue
            for i in range(1, len(tokens)):
                start = max(0, i - WINDOW)
                ctx = tokens[start:i]
                if len(ctx) < WINDOW:
                    ctx = [pad_id] * (WINDOW - len(ctx)) + ctx
                windows_train.append(ctx)
                targets_train.append(tokens[i])
        wt = torch.tensor(windows_train[:50000], dtype=torch.long, device=device)
        tt = torch.tensor(targets_train[:50000], dtype=torch.long, device=device)
        bs = 256
        model.train()
        for ep in range(3):
            perm = torch.randperm(len(wt))
            for i in range(0, len(wt), bs):
                idx = perm[i:i + bs]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(wt[idx]), tt[idx])
                loss.backward()
                optimizer.step()

        sample_tokens = [event_to_token[e] for e in test_sample['EventId'].iloc[0] if e in event_to_token][:WINDOW]
        while len(sample_tokens) < WINDOW:
            sample_tokens = [pad_id] + sample_tokens
        sample_window = torch.tensor([sample_tokens], dtype=torch.long, device=device)

        model.eval()

        @torch.no_grad()
        def deeplog_step():
            logits = model(sample_window)
            _ = logits.topk(9, dim=-1).indices
            if device == 'cuda':
                torch.cuda.synchronize()

        lat = bench(deeplog_step)
        deeplog_results[device] = lat
        print(f'DeepLog ({device}): {lat}', flush=True)

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

    tcn_results = {}
    for device in [DEVICE_CPU, DEVICE_GPU] if DEVICE_GPU == 'cuda' else [DEVICE_CPU]:
        model = TCNModel(vocab_size=vocab_size, pad_id=pad_id).to(device).eval()
        normal_train = train_df[train_df['y'] == 0]
        seqs = normal_train['EventId'].tolist()[:5000]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        windows_train, targets_train = [], []
        for seq in seqs:
            tokens = [event_to_token[e] for e in seq if e in event_to_token]
            if len(tokens) < 2:
                continue
            for i in range(1, len(tokens)):
                start = max(0, i - WINDOW)
                ctx = tokens[start:i]
                if len(ctx) < WINDOW:
                    ctx = [pad_id] * (WINDOW - len(ctx)) + ctx
                windows_train.append(ctx)
                targets_train.append(tokens[i])
        wt = torch.tensor(windows_train[:50000], dtype=torch.long, device=device)
        tt = torch.tensor(targets_train[:50000], dtype=torch.long, device=device)
        bs = 256
        model.train()
        for ep in range(3):
            perm = torch.randperm(len(wt))
            for i in range(0, len(wt), bs):
                idx = perm[i:i + bs]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(wt[idx]), tt[idx])
                loss.backward()
                optimizer.step()

        sample_tokens = [event_to_token[e] for e in test_sample['EventId'].iloc[0] if e in event_to_token][:WINDOW]
        while len(sample_tokens) < WINDOW:
            sample_tokens = [pad_id] + sample_tokens
        sample_window = torch.tensor([sample_tokens], dtype=torch.long, device=device)

        model.eval()

        @torch.no_grad()
        def tcn_step():
            logits = model(sample_window)
            _ = logits.topk(9, dim=-1).indices
            if device == 'cuda':
                torch.cuda.synchronize()

        lat = bench(tcn_step)
        tcn_results[device] = lat
        print(f'TCN ({device}): {lat}', flush=True)

    MAX_SEQ = 64

    class MiniLogBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size + 1, 64, padding_idx=pad_id)
            self.pos = nn.Embedding(MAX_SEQ, 64)
            layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, activation='gelu', batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.cls = nn.Linear(64, vocab_size + 1)

        def forward(self, x, attn):
            b, l = x.shape
            pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
            h = self.enc(self.tok(x) + self.pos(pos), src_key_padding_mask=(attn == 0))
            return self.cls(h)

    transformer_results = {}
    for device in [DEVICE_CPU, DEVICE_GPU] if DEVICE_GPU == 'cuda' else [DEVICE_CPU]:
        model = MiniLogBERT().to(device).eval()
        tokens = [event_to_token[e] for e in test_sample['EventId'].iloc[0] if e in event_to_token][:MAX_SEQ]
        attn = [1] * len(tokens)
        if len(tokens) < MAX_SEQ:
            pad = MAX_SEQ - len(tokens)
            tokens += [pad_id] * pad
            attn += [0] * pad
        sample_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        sample_attn = torch.tensor([attn], dtype=torch.long, device=device)

        @torch.no_grad()
        def tx_step():
            _ = model(sample_tokens, sample_attn)
            if device == 'cuda':
                torch.cuda.synchronize()

        lat = bench(tx_step)
        transformer_results[device] = lat
        print(f'Transformer ({device}): {lat}', flush=True)

    drain_lines = raw['content'].tolist()[:N_BENCH_SAMPLES + WARMUP]
    counter['i'] = 0

    def drain_step():
        miner.add_log_message(drain_lines[counter['i'] % len(drain_lines)])
        counter['i'] += 1

    drain_lat = bench(drain_step, n_samples=500, warmup=20)
    print(f'Drain3 add_log_message: {drain_lat}', flush=True)

    summary = {
        'seed': SEED,
        'n_bench_samples': N_BENCH_SAMPLES,
        'hardware': {
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'drain3_parse': drain_lat,
        'pca': pca_lat,
        'truncated_svd': svd_lat,
        'isolation_forest': if_lat,
        'deeplog': deeplog_results,
        'tcn': tcn_results,
        'transformer': transformer_results,
        'total_seconds': float(time.perf_counter() - t0),
    }
    with open('latency_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nTotal seconds: {summary["total_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
