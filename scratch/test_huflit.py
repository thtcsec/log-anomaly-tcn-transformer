import pandas as pd
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

df = pd.read_csv('thuvien_20260604_094551/access.csv')
print("Total rows:", len(df))
print("Anomalous rows:", (df['suspicious_signals'] != '-').sum())

# Drain3 configuration
config = TemplateMinerConfig()
config.profiling_enabled = False
config.drain_sim_th = 0.5
config.drain_depth = 4
miner = TemplateMiner(config=config)

# Parse logs
parsed = []
for row in df.itertuples(index=False):
    # Combine method and path
    msg = f"{row.method} {row.path}"
    result = miner.add_log_message(msg)
    parsed.append({
        'ip': row.ip,
        'timestamp': row.timestamp,
        'EventId': int(result['cluster_id']),
        'LineAnomaly': int(row.suspicious_signals != '-')
    })
parsed_df = pd.DataFrame(parsed)
print("Unique templates found:", parsed_df['EventId'].nunique())

# Group by IP, sort by timestamp
grouped = []
for ip, group in parsed_df.groupby('ip'):
    group = group.sort_values('timestamp')
    events = group['EventId'].tolist()
    anomalies = group['LineAnomaly'].tolist()
    
    # Slide a window of size 10, step 5
    W = 10
    step = 5
    if len(events) < W:
        # Pad with 0 (dummy event)
        padded_events = events + [0]*(W - len(events))
        grouped.append({
            'ip': ip,
            'EventId': padded_events,
            'y': int(any(anomalies))
        })
    else:
        for i in range(0, len(events) - W + 1, step):
            chunk_events = events[i:i+W]
            chunk_anoms = anomalies[i:i+W]
            grouped.append({
                'ip': ip,
                'EventId': chunk_events,
                'y': int(any(chunk_anoms))
            })

grouped_df = pd.DataFrame(grouped)
print("Total grouped sequences:", len(grouped_df))
print("Anomalous sequences:", grouped_df['y'].sum())
print("Normal sequences:", (grouped_df['y'] == 0).sum())
