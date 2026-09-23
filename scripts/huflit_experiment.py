"""Evaluation on HUFLIT Library Logs.

Groups logs by client IP address, slices into windows, and runs the entire
evaluation pipeline (PCA, TruncatedSVD, Isolation Forest, DeepLog, TCN, and Transformer)
across 5 random seeds.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEEDS = [21, 42, 84, 123, 777]
WINDOW_SIZE = 10
STEP_SIZE = 5
DRAIN_SIM_TH = 0.5
DRAIN_DEPTH = 4

MAX_SEQ_LEN = 10
MASK_PROB = 0.15
TX_BATCH = 128
TX_EPOCHS = 5
TX_LR = 1e-3
D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
FF_DIM = 128
SCORE_PASSES = 5

DL_WINDOW = 5
DL_TOP_K = 3
DL_BATCH = 128
DL_EPOCHS = 8
DL_LR = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, flush=True)


def load_huflit_data():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    default = root / "data" / "careerhub_20260604_095930" / "access.csv"
    csv_path = Path(os.environ.get("HUFLIT_CAREER_CSV", default))
    if not csv_path.exists():
        # backward-compatible local layout
        alt = root / "careerhub_20260604_095930" / "access.csv"
        csv_path = alt if alt.exists() else csv_path
    df = pd.read_csv(csv_path)
    print('Loaded HUFLIT careerhub logs from', csv_path, '| Rows:', len(df), '| Anomalies:', (df['suspicious_signals'] != '-').sum(), flush=True)
    return df


def drain_parse(raw_df):
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = DRAIN_SIM_TH
    config.drain_depth = DRAIN_DEPTH
    miner = TemplateMiner(config=config)
    parsed = []
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc='drain3-huflit'):
        msg = f"{row.method} {row.path}"
        result = miner.add_log_message(msg)
        parsed.append({
            'ip': row.ip,
            'timestamp': row.timestamp,
            'LineAnomaly': int(row.suspicious_signals != '-'),
            'EventId': int(result['cluster_id'])
        })
    return pd.DataFrame(parsed)


def group_by_ip_sequences(parsed_df, window=WINDOW_SIZE, step=STEP_SIZE):
    rows = []
    for ip, group in parsed_df.groupby('ip'):
        group = group.sort_values('timestamp')
        events = group['EventId'].tolist()
        anomalies = group['LineAnomaly'].tolist()
        
        if len(events) < window:
            padded_events = events + [0] * (window - len(events))
            rows.append({
                'GroupKey': ip,
                'EventId': padded_events,
                'SeqLen': len(padded_events),
                'y': int(any(anomalies))
            })
        else:
            for i in range(0, len(events) - window + 1, step):
                chunk_events = events[i:i+window]
                chunk_anoms = anomalies[i:i+window]
                rows.append({
                    'GroupKey': f"{ip}_{i}",
                    'EventId': chunk_events,
                    'SeqLen': int(window),
                    'y': int(any(chunk_anoms))
                })
    out = pd.DataFrame(rows)
    out['text'] = out['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    return out


def run_baselines(data, seed):
    np.random.seed(seed)
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=seed, stratify=data['y'])
    y_test = test_df['y'].values
    out = []

    vec_pca = CountVectorizer()
    X_train = vec_pca.fit_transform(train_df['text']).toarray()
    X_test = vec_pca.transform(test_df['text']).toarray()
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
    out.append({'Method': 'PCA', 'Seed': seed, 'Precision': float(p), 'Recall': float(r), 'F1': float(f1)})

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

    vec_if = TfidfVectorizer()
    normal_train = train_df[train_df['y'] == 0]
    X_if_train = vec_if.fit_transform(normal_train['text'])
    X_if_test = vec_if.transform(test_df['text'])
    # Primary protocol: fit on normals only; threshold -decision_function at 95th
    # percentile of normal training scores (same family as PCA). Do NOT use
    # contamination=train_anomaly_rate as the unsupervised headline (that is
    # label-informed prevalence calibration).
    iso = IsolationForest(
        n_estimators=200, contamination='auto', random_state=seed, n_jobs=-1
    )
    iso.fit(X_if_train)
    train_scores = -iso.decision_function(X_if_train)
    thr = np.percentile(train_scores, 95)
    test_scores = -iso.decision_function(X_if_test)
    y_pred = (test_scores > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    out.append({'Method': 'Isolation Forest', 'Seed': seed, 'Precision': float(p), 'Recall': float(r), 'F1': float(f1)})

    return out


class LogSequenceDataset(Dataset):
    def __init__(self, df, event_to_token, pad_id, max_len=MAX_SEQ_LEN):
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
            pad_len = self.max_len - len(tokens)
            tokens += [self.pad_id] * pad_len
            attention += [0] * pad_len
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'attention': torch.tensor(attention, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MiniLogBERT(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FF_DIM,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.classifier = nn.Linear(D_MODEL, vocab_size)

    def forward(self, input_ids, attention):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        kpm = attention == 0
        h = self.encoder(x, src_key_padding_mask=kpm)
        return self.classifier(h)


def make_masked_batch(tokens, attention, mask_id, pad_id):
    inputs = tokens.clone()
    labels = torch.full_like(tokens, fill_value=-100)
    rand = torch.rand(tokens.shape, device=tokens.device)
    mask = (rand < MASK_PROB) & (attention == 1) & (tokens != pad_id)
    for i in range(tokens.size(0)):
        if attention[i].sum() > 0 and not mask[i].any():
            valid = torch.where((attention[i] == 1) & (tokens[i] != pad_id))[0]
            chosen = torch.randint(len(valid), (1,), device=tokens.device).item()
            mask[i, valid[chosen]] = True
    labels[mask] = tokens[mask]
    inputs[mask] = mask_id
    return inputs, labels


def run_transformer(data, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    mask_id = len(event_to_token) + 1
    vocab_size = mask_id + 1

    train_df, temp_df = train_test_split(data, test_size=0.4, random_state=seed, stratify=data['y'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['y'])
    normal_train_df = train_df[train_df['y'] == 0].copy()

    train_loader = DataLoader(LogSequenceDataset(normal_train_df, event_to_token, pad_id),
                              batch_size=TX_BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(LogSequenceDataset(val_df, event_to_token, pad_id),
                            batch_size=TX_BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(LogSequenceDataset(test_df, event_to_token, pad_id),
                             batch_size=TX_BATCH, shuffle=False, num_workers=0)

    model = MiniLogBERT(vocab_size=vocab_size, max_len=MAX_SEQ_LEN, pad_id=pad_id).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TX_LR, weight_decay=1e-4)

    model.train()
    for epoch in range(1, TX_EPOCHS + 1):
        for batch in train_loader:
            tokens = batch['tokens'].to(DEVICE)
            attention = batch['attention'].to(DEVICE)
            inputs, targets = make_masked_batch(tokens, attention, mask_id, pad_id)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, attention)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def score(loader):
        model.eval()
        nlls = []
        for batch in loader:
            tokens = batch['tokens'].to(DEVICE)
            attention = batch['attention'].to(DEVICE)
            bsz, seq_len = tokens.shape
            batch_nll = torch.zeros(bsz, device=DEVICE)
            for step in range(SCORE_PASSES):
                inputs, targets = make_masked_batch(tokens, attention, mask_id, pad_id)
                logits = model(inputs, attention)
                log_probs = torch.log_softmax(logits, dim=-1)
                mask = targets != -100
                if mask.any():
                    lp = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                    lp = torch.where(mask, -lp, 0.0)
                    batch_nll += lp.sum(dim=-1) / torch.clamp(mask.sum(dim=-1).float(), min=1.0)
            nlls.extend((batch_nll / SCORE_PASSES).cpu().numpy())
        return np.array(nlls)

    val_nll = score(val_loader)
    test_nll = score(test_loader)
    val_labels = val_df['y'].astype(int).values
    test_labels = test_df['y'].astype(int).values

    normal_val = val_nll[val_labels == 0]
    th = np.percentile(normal_val, 95) if len(normal_val) > 0 else 0.0
    pred = (test_nll > th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, pred, average='binary', zero_division=0)
    return [{'Method': 'Transformer (val-tuned)', 'Seed': seed,
             'Precision': float(p), 'Recall': float(r), 'F1': float(f1)}]


class WindowDataset(Dataset):
    def __init__(self, sequences, event_to_token, pad_id, window=DL_WINDOW):
        self.windows = []
        self.targets = []
        self.seq_idx = []
        for s_idx, seq in enumerate(sequences):
            tokens = [event_to_token[e] for e in seq if e in event_to_token]
            if len(tokens) < window + 1:
                tokens = tokens + [pad_id] * (window + 1 - len(tokens))
            for i in range(len(tokens) - window):
                self.windows.append(tokens[i : i + window])
                self.targets.append(tokens[i + window])
                self.seq_idx.append(s_idx)
        self.windows = np.array(self.windows, dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64)
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
        seq_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_nll = np.zeros(int(n_seq), dtype=np.float64)
        loader = DataLoader(ds, batch_size=DL_BATCH, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            sidx = batch['seq_idx'].numpy()
            logits = model(windows)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = logits.topk(min(DL_TOP_K, vocab_size), dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            tgt_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            for s, m, lp in zip(sidx, mismatch, tgt_lp):
                seq_mismatch[s] += int(m)
                seq_count[s] += 1
                seq_nll[s] += float(-lp)
        seq_count = np.maximum(seq_count, 1)
        return (seq_mismatch > 0).astype(int), seq_nll / seq_count

    val_binary, val_nll = score(val_ds)
    test_binary, test_nll = score(test_ds)
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)
    
    nv = val_nll[val_labels == 0]
    th = np.percentile(nv, 95) if len(nv) > 0 else 0.0
    pred_n = (test_nll > th).astype(int)
    p_n, r_n, f1_n, _ = precision_recall_fscore_support(test_labels, pred_n, average='binary', zero_division=0)

    return [
        {'Method': 'DeepLog (top-k)', 'Seed': seed, 'Precision': float(p_b), 'Recall': float(r_b), 'F1': float(f1_b)},
        {'Method': 'DeepLog (NLL)', 'Seed': seed, 'Precision': float(p_n), 'Recall': float(r_n), 'F1': float(f1_n)},
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
        seq_count = np.zeros(int(n_seq), dtype=np.int64)
        seq_nll = np.zeros(int(n_seq), dtype=np.float64)
        loader = DataLoader(ds, batch_size=DL_BATCH, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            sidx = batch['seq_idx'].numpy()
            logits = model(windows)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = logits.topk(min(DL_TOP_K, vocab_size), dim=-1).indices
            mismatch = (topk != targets.unsqueeze(1)).all(dim=-1).cpu().numpy()
            tgt_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            for s, m, lp in zip(sidx, mismatch, tgt_lp):
                seq_mismatch[s] += int(m)
                seq_count[s] += 1
                seq_nll[s] += float(-lp)
        seq_count = np.maximum(seq_count, 1)
        return (seq_mismatch > 0).astype(int), seq_nll / seq_count

    val_binary, val_nll = score(val_ds)
    test_binary, test_nll = score(test_ds)
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(test_labels, test_binary, average='binary', zero_division=0)
    
    nv = val_nll[val_labels == 0]
    th = np.percentile(nv, 95) if len(nv) > 0 else 0.0
    pred_n = (test_nll > th).astype(int)
    p_n, r_n, f1_n, _ = precision_recall_fscore_support(test_labels, pred_n, average='binary', zero_division=0)

    return [
        {'Method': 'TCN (top-k)', 'Seed': seed, 'Precision': float(p_b), 'Recall': float(r_b), 'F1': float(f1_b)},
        {'Method': 'TCN (NLL)', 'Seed': seed, 'Precision': float(p_n), 'Recall': float(r_n), 'F1': float(f1_n)},
    ]


def main():
    t0 = time.perf_counter()
    raw_df = load_huflit_data()
    events = drain_parse(raw_df)
    print(f'parsed logs: {len(events)}, templates: {events["EventId"].nunique()}', flush=True)
    data = group_by_ip_sequences(events, window=WINDOW_SIZE, step=STEP_SIZE)
    print(f'grouped sequences: {len(data)} (normal={(data.y == 0).sum()}, anomaly={(data.y == 1).sum()})', flush=True)

    all_rows = []
    for seed in SEEDS:
        print(f'\n=== seed {seed} baselines ===', flush=True)
        rows = run_baselines(data, seed)
        for r in rows:
            print(f"  {r['Method']:<28} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows)
        
    for seed in SEEDS:
        print(f'\n=== seed {seed} transformer ===', flush=True)
        rows = run_transformer(data, seed)
        for r in rows:
            print(f"  {r['Method']:<32} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows)
        
    for seed in SEEDS:
        print(f'\n=== seed {seed} deeplog ===', flush=True)
        rows = run_deeplog(data, seed)
        for r in rows:
            print(f"  {r['Method']:<32} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows)
        
    for seed in SEEDS:
        print(f'\n=== seed {seed} tcn ===', flush=True)
        rows = run_tcn(data, seed)
        for r in rows:
            print(f"  {r['Method']:<32} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}", flush=True)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('huflit_results.csv', index=False)

    summary = {
        'config': {'seeds': SEEDS, 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE},
        'dataset': {
            'rows_loaded': int(len(raw_df)),
            'event_templates': int(events['EventId'].nunique()),
            'sequences': int(len(data)),
            'normal': int((data['y'] == 0).sum()),
            'anomaly': int((data['y'] == 1).sum()),
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
    with open('huflit_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n=== aggregated ===', flush=True)
    agg = df.groupby('Method')[['Precision', 'Recall', 'F1']].agg(['mean', 'std']).round(4)
    print(agg, flush=True)
    print(f'\nTotal seconds: {summary["total_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
