import pandas as pd
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

df = pd.read_csv('careerhub_20260604_095930/access.csv')

config = TemplateMinerConfig()
config.profiling_enabled = False
config.drain_sim_th = 0.5
config.drain_depth = 4
miner = TemplateMiner(config=config)

parsed = []
for row in df.itertuples(index=False):
    msg = f"{row.method} {row.path}"
    result = miner.add_log_message(msg)
    parsed.append({
        'ip': row.ip,
        'timestamp': row.timestamp,
        'LineAnomaly': int(row.suspicious_signals != '-'),
        'EventId': int(result['cluster_id'])
    })
parsed_df = pd.DataFrame(parsed)

# Group by IP into sequences
ip_seqs = []
for ip, group in parsed_df.groupby('ip'):
    group = group.sort_values('timestamp')
    events = group['EventId'].tolist()
    anomalies = group['LineAnomaly'].tolist()
    
    W = 10
    step = 5
    seqs = []
    if len(events) < W:
        padded_events = events + [0] * (W - len(events))
        seqs.append({
            'EventId': padded_events,
            'y': int(any(anomalies))
        })
    else:
        for i in range(0, len(events) - W + 1, step):
            chunk_events = events[i:i+W]
            chunk_anoms = anomalies[i:i+W]
            seqs.append({
                'EventId': chunk_events,
                'y': int(any(chunk_anoms))
            })
    ip_seqs.append({
        'ip': ip,
        'seqs': seqs,
        'has_anomaly': int(any(anomalies))
    })

# Split IPs: 70% train, 30% test. Stratify by whether the IP has any anomaly.
ips_df = pd.DataFrame(ip_seqs)
train_ips, test_ips = train_test_split(ips_df, test_size=0.3, random_state=42, stratify=ips_df['has_anomaly'])

# Flatten train and test sequences
train_rows = []
for idx, row in train_ips.iterrows():
    for seq in row['seqs']:
        train_rows.append({
            'text': ' '.join([f'E{x}' for x in seq['EventId']]),
            'y': seq['y']
        })
train_df = pd.DataFrame(train_rows)

test_rows = []
for idx, row in test_ips.iterrows():
    for seq in row['seqs']:
        test_rows.append({
            'text': ' '.join([f'E{x}' for x in seq['EventId']]),
            'y': seq['y']
        })
test_df = pd.DataFrame(test_rows)

print("Train sequences:", len(train_df), "with anomalies:", train_df['y'].sum())
print("Test sequences:", len(test_df), "with anomalies:", test_df['y'].sum())

# Run PCA
vec = CountVectorizer()
X_train = vec.fit_transform(train_df['text']).toarray()
X_test = vec.transform(test_df['text']).toarray()
normal_mask = train_df['y'].values == 0

scaler = StandardScaler()
X_normal = scaler.fit_transform(X_train[normal_mask])
X_test_scaled = scaler.transform(X_test)

n_components = max(1, min(20, X_normal.shape[1], X_normal.shape[0] - 1))
pca = PCA(n_components=n_components, random_state=42)
pca.fit(X_normal)

X_train_recon = pca.inverse_transform(pca.transform(X_normal))
train_err = np.mean((X_normal - X_train_recon) ** 2, axis=1)
threshold = np.percentile(train_err, 95)

X_test_recon = pca.inverse_transform(pca.transform(X_test_scaled))
test_err = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)
y_pred = (test_err > threshold).astype(int)

p, r, f1, _ = precision_recall_fscore_support(test_df['y'].values, y_pred, average='binary', zero_division=0)
print(f"IP-disjoint PCA results: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
