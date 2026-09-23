"""Regenerate ROC/PR figures with ALL 6 methods for vnict2026.tex.

Includes: PCA, TruncatedSVD, Isolation Forest, DeepLog (NLL), TCN (NLL), Transformer (masked loss).
Uses seed 42 on 500k HDFS rows.
"""

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

WINDOW = 10
TOP_K = 9
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

# Transformer config
TF_D_MODEL = 64
TF_NHEAD = 4
TF_LAYERS = 2
TF_FF = 128
TF_MASK_RATIO = 0.15
TF_EPOCHS = 5
TF_MAX_LEN = 64

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, flush=True)


# ─── Data loading & Drain3 ───────────────────────────────────────────

def load_and_parse():
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
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df)):
        result = miner.add_log_message(row.content)
        parsed.append((row.BlockId, int(row.LineAnomaly), int(result['cluster_id'])))
    events = pd.DataFrame(parsed, columns=['BlockId', 'LineAnomaly', 'EventId'])
    print(f'Templates: {events["EventId"].nunique()}', flush=True)

    data = (
        events.groupby('BlockId')
        .agg(EventId=('EventId', list), SeqLen=('EventId', 'size'), y=('LineAnomaly', 'max'))
        .reset_index()
    )
    data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
    print(f'Blocks: {len(data)} (normal={int((data.y==0).sum())}, anomaly={int((data.y==1).sum())})', flush=True)
    return data


# ─── Baseline scores ─────────────────────────────────────────────────

def compute_baseline_scores(train_df, test_df):
    normal_mask = train_df['y'].values == 0

    # PCA
    vec_pca = CountVectorizer()
    X_train_all = vec_pca.fit_transform(train_df['text']).toarray()
    X_test_all = vec_pca.transform(test_df['text']).toarray()
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_train_all[normal_mask])
    X_test_scaled = scaler.transform(X_test_all)
    n_comp = max(1, min(20, X_normal.shape[1], X_normal.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(X_normal)
    X_recon = pca.inverse_transform(pca.transform(X_test_scaled))
    pca_score = np.mean((X_test_scaled - X_recon) ** 2, axis=1)

    # TruncatedSVD
    vec_svd = CountVectorizer()
    X_train_sp = vec_svd.fit_transform(train_df['text'])
    X_test_sp = vec_svd.transform(test_df['text'])
    X_normal_sp = X_train_sp[normal_mask]
    n_comp_svd = max(1, min(20, X_normal_sp.shape[1] - 1, X_normal_sp.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp_svd, random_state=SEED)
    svd.fit(X_normal_sp)
    Xh = svd.inverse_transform(svd.transform(X_test_sp))
    svd_score = np.mean((X_test_sp.toarray() - Xh) ** 2, axis=1)

    # Isolation Forest
    vec_if = TfidfVectorizer()
    normal_train = train_df[train_df['y'] == 0]
    X_if_train = vec_if.fit_transform(normal_train['text'])
    X_if_test = vec_if.transform(test_df['text'])
    contamination = max(0.001, min(0.2, train_df['y'].mean()))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1)
    iso.fit(X_if_train)
    if_score = -iso.decision_function(X_if_test)

    return {
        'PCA': pca_score,
        'TruncatedSVD': svd_score,
        'Isolation Forest': if_score,
    }


# ─── DeepLog / TCN models ────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, sequences, event_to_token, pad_id, window=WINDOW):
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
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS,
                            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, vocab_size)

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
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)
        self.tcn = TemporalConvNet(EMBED_DIM, [64, 64, 64], kernel_size=2, dropout=DROPOUT)
        self.classifier = nn.Linear(64, vocab_size)

    def forward(self, windows):
        x = self.embedding(windows).transpose(1, 2)
        y = self.tcn(x)
        return self.classifier(y[:, :, -1])


def train_and_score_nextev(ModelClass, model_name, data, train_df, test_df):
    """Train a next-event model (LSTM or TCN), return continuous NLL scores for test set."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1

    normal_train_df = train_df[train_df['y'] == 0]
    train_ds = WindowDataset(normal_train_df['EventId'].tolist(), event_to_token, pad_id)
    test_ds = WindowDataset(test_df['EventId'].tolist(), event_to_token, pad_id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ModelClass(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print(f'  Training {model_name}... params={sum(p.numel() for p in model.parameters())}', flush=True)

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
        if epoch == 1 or epoch == EPOCHS:
            print(f'    epoch {epoch}: loss={np.mean(losses):.4f}', flush=True)

    # Score: continuous NLL per sequence
    @torch.no_grad()
    def get_nll_scores(ds):
        model.eval()
        n_seq = int(ds.seq_idx.max()) + 1 if len(ds) > 0 else 0
        seq_nll_sum = np.zeros(n_seq, dtype=np.float64)
        seq_window_count = np.zeros(n_seq, dtype=np.int64)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for batch in loader:
            windows = batch['window'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            seq_idx = batch['seq_idx'].numpy()
            logits = model(windows)
            log_probs = torch.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            for s, lp in zip(seq_idx, target_lp):
                seq_nll_sum[s] += float(-lp)
                seq_window_count[s] += 1
        seq_window_count = np.maximum(seq_window_count, 1)
        return seq_nll_sum / seq_window_count

    return get_nll_scores(test_ds)


# ─── Transformer masked event model ──────────────────────────────────

class TransformerMaskedEvent(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=TF_D_MODEL, nhead=TF_NHEAD,
                 num_layers=TF_LAYERS, dim_ff=TF_FF, max_len=TF_MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = vocab_size  # use extra token for [MASK]
        self.vocab_out = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_ff, dropout=dropout,
                                                    activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, src_key_padding_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(x)


class TransformerSeqDataset(Dataset):
    def __init__(self, sequences, event_to_token, pad_id, max_len=TF_MAX_LEN):
        self.pad_id = pad_id
        self.seqs = []
        for seq in sequences:
            tokens = [event_to_token[e] for e in seq if e in event_to_token]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            self.seqs.append(tokens)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def tf_collate(batch, pad_id, max_len=TF_MAX_LEN):
    lengths = [len(s) for s in batch]
    ml = min(max(lengths), max_len)
    input_ids = torch.full((len(batch), ml), pad_id, dtype=torch.long)
    for i, s in enumerate(batch):
        l = min(len(s), ml)
        input_ids[i, :l] = torch.tensor(s[:l], dtype=torch.long)
    return input_ids


def train_and_score_transformer(data, train_df, test_df):
    """Train Transformer masked event model, return continuous masked loss scores for test set."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    all_event_ids = sorted({e for seq in data['EventId'] for e in seq})
    event_to_token = {eid: i + 1 for i, eid in enumerate(all_event_ids)}
    pad_id = 0
    vocab_size = len(event_to_token) + 1

    normal_train_df = train_df[train_df['y'] == 0]
    train_ds = TransformerSeqDataset(normal_train_df['EventId'].tolist(), event_to_token, pad_id)
    test_ds = TransformerSeqDataset(test_df['EventId'].tolist(), event_to_token, pad_id)

    from functools import partial
    collate_fn = partial(tf_collate, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = TransformerMaskedEvent(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    mask_id = model.mask_id
    print(f'  Training Transformer... params={sum(p.numel() for p in model.parameters())}', flush=True)

    model.train()
    for epoch in range(1, TF_EPOCHS + 1):
        losses = []
        for input_ids in train_loader:
            input_ids = input_ids.to(DEVICE)
            # Create masked input
            targets = input_ids.clone()
            mask_prob = torch.rand_like(input_ids, dtype=torch.float)
            mask_positions = (mask_prob < TF_MASK_RATIO) & (input_ids != pad_id)
            masked_input = input_ids.clone()
            masked_input[mask_positions] = mask_id
            targets[~mask_positions] = pad_id  # only compute loss on masked positions

            padding_mask = (masked_input == pad_id)
            logits = model(masked_input, src_key_padding_mask=padding_mask)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == TF_EPOCHS:
            print(f'    epoch {epoch}: loss={np.mean(losses):.4f}', flush=True)

    # Score: average masked loss over N_MASKS random masks per sequence
    N_MASKS = 5

    @torch.no_grad()
    def get_masked_scores(ds):
        model.eval()
        all_scores = np.zeros(len(ds), dtype=np.float64)
        test_loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)
        for mask_iter in range(N_MASKS):
            torch.manual_seed(SEED + mask_iter)
            offset = 0
            for input_ids in test_loader:
                input_ids = input_ids.to(DEVICE)
                bs = input_ids.shape[0]
                targets = input_ids.clone()
                mask_prob = torch.rand_like(input_ids, dtype=torch.float)
                mask_positions = (mask_prob < TF_MASK_RATIO) & (input_ids != pad_id)
                # Ensure at least 1 mask per sequence
                for b in range(bs):
                    if not mask_positions[b].any():
                        valid = (input_ids[b] != pad_id).nonzero(as_tuple=True)[0]
                        if len(valid) > 0:
                            mask_positions[b, valid[0]] = True

                masked_input = input_ids.clone()
                masked_input[mask_positions] = mask_id
                targets_masked = input_ids.clone()
                targets_masked[~mask_positions] = pad_id

                padding_mask = (masked_input == pad_id)
                logits = model(masked_input, src_key_padding_mask=padding_mask)
                # per-sample loss
                for b in range(bs):
                    mp = mask_positions[b]
                    if mp.any():
                        sample_loss = nn.functional.cross_entropy(
                            logits[b][mp], input_ids[b][mp], reduction='mean'
                        )
                        all_scores[offset + b] += float(sample_loss.cpu())
                    else:
                        all_scores[offset + b] += 0.0
                offset += bs
        return all_scores / N_MASKS

    return get_masked_scores(test_ds)


# ─── Plotting ─────────────────────────────────────────────────────────

COLORS = {
    'PCA': '#1f77b4',
    'TruncatedSVD': '#ff7f0e',
    'Isolation Forest': '#2ca02c',
    'DeepLog (NLL)': '#d62728',
    'TCN (NLL)': '#9467bd',
    'Transformer': '#8c564b',
}
LINESTYLES = {
    'PCA': '-',
    'TruncatedSVD': '-',
    'Isolation Forest': '-',
    'DeepLog (NLL)': '--',
    'TCN (NLL)': '--',
    'Transformer': ':',
}


def plot_roc(score_dict, y_true, output_path='images/roc_curves.png'):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=COLORS.get(name),
                linestyle=LINESTYLES.get(name, '-'),
                label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curves on HDFS test set (seed 42)')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {output_path}', flush=True)


def plot_pr(score_dict, y_true, output_path='images/pr_curves.png'):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    for name, scores in score_dict.items():
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec_curve, prec_curve, linewidth=2, color=COLORS.get(name),
                linestyle=LINESTYLES.get(name, '-'),
                label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curves on HDFS test set (seed 42)')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {output_path}', flush=True)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    data = load_and_parse()

    # Use same 70/30 split as baselines
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=SEED, stratify=data['y'])
    y_true = test_df['y'].values

    print('\n--- Computing baseline scores ---', flush=True)
    score_dict = compute_baseline_scores(train_df, test_df)

    print('\n--- Training DeepLog (LSTM) ---', flush=True)
    dl_nll = train_and_score_nextev(DeepLogLSTM, 'DeepLog', data, train_df, test_df)
    score_dict['DeepLog (NLL)'] = dl_nll

    print('\n--- Training TCN ---', flush=True)
    tcn_nll = train_and_score_nextev(TCNModel, 'TCN', data, train_df, test_df)
    score_dict['TCN (NLL)'] = tcn_nll

    print('\n--- Training Transformer ---', flush=True)
    tf_scores = train_and_score_transformer(data, train_df, test_df)
    score_dict['Transformer'] = tf_scores

    print('\n--- Plotting ---', flush=True)
    plot_roc(score_dict, y_true)
    plot_pr(score_dict, y_true)

    elapsed = time.perf_counter() - t0
    print(f'\nDone! Total time: {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
