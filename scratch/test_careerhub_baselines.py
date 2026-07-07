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
print("Careerhub rows:", len(df))
print("Anomalous rows:", (df['suspicious_signals'] != '-').sum())

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
print("Unique templates:", parsed_df['EventId'].nunique())

# Group by IP
rows = []
for ip, group in parsed_df.groupby('ip'):
    group = group.sort_values('timestamp')
    events = group['EventId'].tolist()
    anomalies = group['LineAnomaly'].tolist()
    
    W = 10
    step = 5
    if len(events) < W:
        padded_events = events + [0] * (W - len(events))
        rows.append({
            'GroupKey': ip,
            'EventId': padded_events,
            'y': int(any(anomalies))
        })
    else:
        for i in range(0, len(events) - W + 1, step):
            chunk_events = events[i:i+W]
            chunk_anoms = anomalies[i:i+W]
            rows.append({
                'GroupKey': f"{ip}_{i}",
                'EventId': chunk_events,
                'y': int(any(chunk_anoms))
            })
data = pd.DataFrame(rows)
data['text'] = data['EventId'].apply(lambda xs: ' '.join([f'E{x}' for x in xs]))
print("Grouped sequences:", len(data))
print("Anomalous sequences:", data['y'].sum())

# Run PCA baseline on seed 42
train_df, test_df = train_test_split(data, test_size=0.3, random_state=42, stratify=data['y'])
y_test = test_df['y'].values

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

p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
print(f"PCA results: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
