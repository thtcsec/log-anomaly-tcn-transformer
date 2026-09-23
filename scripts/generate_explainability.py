"""Generate explainability case studies for HDFS and BGL.
Trains a quick TCN on normal logs, runs on a test anomaly sequence,
and extracts the failed position, actual event, and top-3 predicted events.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DL_WINDOW = 10
D_MODEL = 64
N_LAYERS = 2

class WindowDataset(Dataset):
    def __init__(self, seq_list, labels_list, event_to_token, pad_id, window=DL_WINDOW):
        self.window = window
        self.pad_id = pad_id
        self.windows = []
        self.targets = []
        self.seq_idx = []
        self.seq_labels = []
        for s_idx, (seq, label) in enumerate(zip(seq_list, labels_list)):
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
                self.seq_labels.append(label)
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

def get_hdfs_example():
    print('--- Loading HDFS rows for explainability example ---')
    ds = load_dataset('logfit-project/HDFS_v1', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= 100000:
            break
        rows.append({
            'content': row['content'],
            'BlockId': row['block_id'],
            'anomaly': int(row['anomaly']),
        })
    df = pd.DataFrame(rows)
    
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5
    miner = TemplateMiner(config=cfg)
    
    parsed = []
    templates = {}
    for r in df.itertuples():
        res = miner.add_log_message(r.content)
        cid = int(res['cluster_id'])
        templates[cid] = res['template_mined']
        parsed.append((r.BlockId, r.anomaly, cid))
    events = pd.DataFrame(parsed, columns=['BlockId', 'anomaly', 'EventId'])
    
    data = events.groupby('BlockId').agg(
        EventId=('EventId', list),
        y=('anomaly', 'max')
    ).reset_index()
    
    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    token_to_event = {i: eid for eid, i in event_to_token.items()}
    pad_id = 0
    vocab_size = len(event_to_token) + 1
    
    train_df, test_df = train_test_split(data, test_size=0.4, random_state=42, stratify=data['y'])
    normal_train = train_df[train_df['y'] == 0]
    
    train_ds = WindowDataset(normal_train['EventId'].tolist(), normal_train['y'].tolist(), event_to_token, pad_id)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = TCNModel(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        for b in train_loader:
            w = b['window'].to(DEVICE)
            t = b['target'].to(DEVICE)
            optimizer.zero_grad()
            l = criterion(model(w), t)
            l.backward()
            optimizer.step()
            
    # Find an anomalous block with mismatch
    model.eval()
    anom_test = test_df[test_df['y'] == 1]
    
    for idx, row in anom_test.iterrows():
        seq = row['EventId']
        tokens = [event_to_token[e] for e in seq if e in event_to_token]
        if len(tokens) < 5:
            continue
        
        # Test positions
        for i in range(1, len(tokens)):
            start = max(0, i - DL_WINDOW)
            ctx = tokens[start:i]
            if len(ctx) < DL_WINDOW:
                ctx = [pad_id] * (DL_WINDOW - len(ctx)) + ctx
            
            with torch.no_grad():
                logits = model(torch.tensor([ctx], device=DEVICE))
                probs = torch.softmax(logits, dim=-1)[0]
                topk = probs.topk(5)
                topk_indices = topk.indices.cpu().numpy()
                topk_probs = topk.values.cpu().numpy()
                
            tgt = tokens[i]
            if tgt not in topk_indices[:3]: # mismatch at top-3
                actual_eid = token_to_event[tgt]
                actual_tmpl = templates[actual_eid]
                top3_eids = [token_to_event[tok] if tok in token_to_event else 0 for tok in topk_indices[:3]]
                top3_tmpls = [templates[eid] if eid in templates else 'PAD' for eid in top3_eids]
                nll = -np.log(probs[tgt].item() + 1e-9)
                
                print(f"HDFS Anomaly block: {row['BlockId']}")
                print(f"Failed Position: {i}")
                print(f"Actual: E{actual_eid} ({actual_tmpl})")
                print(f"Top-3 predicted: {top3_eids} (probs: {topk_probs[:3]})")
                print(f"Score (NLL): {nll:.2f}")
                return {
                    'dataset': 'HDFS',
                    'id': str(row['BlockId']),
                    'pos': i,
                    'actual': f"E{actual_eid}: {actual_tmpl[:50]}...",
                    'top3': ', '.join(f"E{e}" for e in top3_eids),
                    'nll': f"{nll:.2f}",
                    'interpretation': 'unusual replication transition / write failure'
                }
    return None

def get_bgl_example():
    print('--- Loading BGL rows for explainability example ---')
    ds = load_dataset('logfit-project/BGL', split='train', streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= 100000:
            break
        rows.append({
            'content': row.get('content', row.get('Content', '')),
            'anomaly': int(row.get('anomaly', 0)),
        })
    df = pd.DataFrame(rows)
    
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5
    miner = TemplateMiner(config=cfg)
    
    parsed = []
    templates = {}
    for r in df.itertuples():
        res = miner.add_log_message(r.content)
        cid = int(res['cluster_id'])
        templates[cid] = res['template_mined']
        parsed.append((r.anomaly, cid))
    events = pd.DataFrame(parsed, columns=['LineAnomaly', 'EventId'])
    
    # group by W=100
    rows_grouped = []
    window = 100
    for start in range(0, len(events) - window + 1, window):
        chunk = events.iloc[start:start + window]
        rows_grouped.append({
            'GroupKey': f'W{start}',
            'EventId': chunk['EventId'].tolist(),
            'y': int(chunk['LineAnomaly'].max()),
        })
    data = pd.DataFrame(rows_grouped)
    
    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    token_to_event = {i: eid for eid, i in event_to_token.items()}
    pad_id = 0
    vocab_size = len(event_to_token) + 1
    
    train_df, test_df = train_test_split(data, test_size=0.4, random_state=42, stratify=data['y'])
    normal_train = train_df[train_df['y'] == 0]
    
    train_ds = WindowDataset(normal_train['EventId'].tolist(), normal_train['y'].tolist(), event_to_token, pad_id)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = TCNModel(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        for b in train_loader:
            w = b['window'].to(DEVICE)
            t = b['target'].to(DEVICE)
            optimizer.zero_grad()
            l = criterion(model(w), t)
            l.backward()
            optimizer.step()
            
    # Find BGL anomaly sequence with mismatch
    model.eval()
    anom_test = test_df[test_df['y'] == 1]
    
    for idx, row in anom_test.iterrows():
        seq = row['EventId']
        tokens = [event_to_token[e] for e in seq if e in event_to_token]
        if len(tokens) < 15:
            continue
        
        # Test positions
        for i in range(1, len(tokens)):
            start = max(0, i - DL_WINDOW)
            ctx = tokens[start:i]
            if len(ctx) < DL_WINDOW:
                ctx = [pad_id] * (DL_WINDOW - len(ctx)) + ctx
            
            with torch.no_grad():
                logits = model(torch.tensor([ctx], device=DEVICE))
                probs = torch.softmax(logits, dim=-1)[0]
                topk = probs.topk(5)
                topk_indices = topk.indices.cpu().numpy()
                topk_probs = topk.values.cpu().numpy()
                
            tgt = tokens[i]
            if tgt not in topk_indices[:3]: # mismatch
                actual_eid = token_to_event[tgt]
                actual_tmpl = templates[actual_eid]
                top3_eids = [token_to_event[tok] if tok in token_to_event else 0 for tok in topk_indices[:3]]
                top3_tmpls = [templates[eid] if eid in templates else 'PAD' for eid in top3_eids]
                nll = -np.log(probs[tgt].item() + 1e-9)
                
                print(f"BGL Anomaly window: {row['GroupKey']}")
                print(f"Failed Position: {i}")
                print(f"Actual: E{actual_eid} ({actual_tmpl})")
                print(f"Top-3 predicted: {top3_eids} (probs: {topk_probs[:3]})")
                print(f"Score (NLL): {nll:.2f}")
                return {
                    'dataset': 'BGL',
                    'id': str(row['GroupKey']),
                    'pos': i,
                    'actual': f"E{actual_eid}: {actual_tmpl[:50]}...",
                    'top3': ', '.join(f"E{e}" for e in top3_eids),
                    'nll': f"{nll:.2f}",
                    'interpretation': 'unusual system/CPU frequency warning'
                }
    return None

def main():
    hdfs_ex = get_hdfs_example()
    bgl_ex = get_bgl_example()
    
    examples = []
    if hdfs_ex:
        examples.append(hdfs_ex)
    if bgl_ex:
        examples.append(bgl_ex)
        
    with open('explainability_cases.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print("\nSaved qualitative cases to explainability_cases.json")

if __name__ == '__main__':
    main()
