"""Statistical significance tests for VNICT2026 paper.

Loads per-seed F1 from multi_seed_results.csv, deeplog_results.csv,
bgl_results.csv. Computes:
  (1) Paired Wilcoxon signed-rank tests between key method pairs.
  (2) Bootstrap 95% CI for F1 of each method (resample with replacement
      across seeds and within-seed test-set replicates by jackknife approx).
  Since we only have 5 seed-level F1 values per (method, dataset), we
  use those 5 values as the population and report Wilcoxon p-values plus
  bootstrap-CI percentile (1000 resamples of size 5 with replacement).
"""
import json
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(42)

HDFS_BASE = pd.read_csv('multi_seed_results.csv')
HDFS_DEEP = pd.read_csv('deeplog_results.csv')
BGL = pd.read_csv('bgl_results.csv')

HDFS = pd.concat([HDFS_BASE, HDFS_DEEP], ignore_index=True)


def per_seed(df, method):
    sub = df[df['Method'] == method].sort_values('Seed')
    return sub['F1'].to_numpy()


def boot_ci(values, n=1000, alpha=0.05):
    rs = RNG.choice(values, size=(n, len(values)), replace=True)
    means = rs.mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(values.std(ddof=1)), float(lo), float(hi)


def wilcoxon_pair(a, b):
    diff = a - b
    if np.all(diff == 0):
        return float('nan'), float('nan')
    stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    return float(stat), float(p)


datasets = {
    'HDFS': (HDFS, ['PCA', 'TruncatedSVD', 'Isolation Forest',
                    'Transformer (unsup. threshold)', 'Transformer (val-tuned threshold)',
                    'DeepLog (top-k binary)', 'DeepLog (NLL threshold)']),
    'BGL': (BGL, ['PCA', 'TruncatedSVD', 'Isolation Forest',
                  'Transformer (unsup. threshold)', 'Transformer (val-tuned threshold)',
                  'DeepLog (top-k binary)', 'DeepLog (NLL threshold)']),
}

results = {'bootstrap_ci': {}, 'wilcoxon': {}, 'n_seeds': 5, 'n_bootstrap': 1000}

for ds_name, (df, methods) in datasets.items():
    results['bootstrap_ci'][ds_name] = {}
    for m in methods:
        v = per_seed(df, m)
        if len(v) == 0:
            continue
        mean, std, lo, hi = boot_ci(v)
        results['bootstrap_ci'][ds_name][m] = {
            'mean': mean, 'std': std, 'ci95_lo': lo, 'ci95_hi': hi,
            'n_seeds': int(len(v)),
        }
        print(f'[{ds_name}] {m:35s} F1 = {mean:.4f} ± {std:.4f}  '
              f'95% CI [{lo:.4f}, {hi:.4f}]')

pairs = [
    ('PCA', 'TruncatedSVD'),
    ('PCA', 'DeepLog (top-k binary)'),
    ('PCA', 'Transformer (val-tuned threshold)'),
    ('DeepLog (top-k binary)', 'Transformer (val-tuned threshold)'),
    ('DeepLog (top-k binary)', 'DeepLog (NLL threshold)'),
    ('TruncatedSVD', 'Isolation Forest'),
]

print('\n=== Paired Wilcoxon signed-rank tests ===')
for ds_name, (df, _) in datasets.items():
    results['wilcoxon'][ds_name] = {}
    for a, b in pairs:
        va = per_seed(df, a)
        vb = per_seed(df, b)
        if len(va) != len(vb) or len(va) == 0:
            continue
        stat, p = wilcoxon_pair(va, vb)
        delta = float(va.mean() - vb.mean())
        results['wilcoxon'][ds_name][f'{a} vs {b}'] = {
            'mean_diff_F1': delta, 'wilcoxon_stat': stat, 'p_value': p,
            'n_seeds': int(len(va)),
        }
        star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
        print(f'[{ds_name}] {a:35s} vs {b:35s} '
              f'ΔF1 = {delta:+.4f}  W = {stat}  p = {p:.4f}  {star}')

with open('stats_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\nSaved -> stats_results.json')
