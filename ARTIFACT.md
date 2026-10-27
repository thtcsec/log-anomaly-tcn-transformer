# Artifact evaluation guide

**Paper:** *Network-Aware Event Sequence Modeling for User-Behavior and System Log Anomaly Detection*

**Authors (LNCS author block):** Thanh Tien Cao, Tu Hoang Trinh, and Ha Manh Tran

**Venue:** CSoNet 2026 (HCMUT, Ho Chi Minh City, 16–18 November 2026). Accepted; Springer LNCS forthcoming. No DOI is claimed.

This artifact is an **applied evaluation study**. It does not claim a new end-to-end detector that wins on every corpus.

## Claims supported by this artifact

1. Drain3 templates are treated as vertices of a directed transition graph; Block ID, time window, or client-IP session is a walk on that graph.
2. GraphWalk scores walks with Laplace-smoothed first-order transitions over \(V \cup \{\mathrm{UNK}\}\), alongside PCA/SVD, Isolation Forest, a DeepLog-style LSTM, TCN, and a masked Transformer.
3. Headline dumps under `logs/` match the manuscript tables for the prefix sizes stated in the paper: HDFS classical 500k / neural+GraphWalk 200k; BGL classical+deep 500k / GraphWalk+grouping 200k.
4. HUFLIT-Career uses client-IP disjoint splits before windowing for classical/GraphWalk headline rows. Historical HUFLIT deep rows live under `logs/legacy_contamination/` and are **not** comparable to the current classical/GraphWalk protocol.
5. HUFLIT labels are heuristic scanner-probe rules, not adjudicated SOC ground truth.

No campus-archive proxy-label validity protocol, no masked TCN–Transformer PLL study, and no CICIDS2017/InSDN feature-reconstruction study is claimed here. Those are separate manuscripts.

## Frozen dumps (no third-party logs required)

Inspect `logs/` (JSON metric dumps committed with the manuscript):

- `graph_transition_results.json` — GraphWalk HDFS/BGL/HUFLIT
- `if_percentile_hdfs_bgl.json` — Isolation Forest 95th-percentile protocol
- `huflit_baselines_nopad.json`, `huflit_graph_ablation.json`
- `bgl_grouping_results.json`, latency dumps

Do not treat files under `logs/legacy_contamination/` as headline numbers.

## Evaluation steps

1. Install `requirements.txt` in a fresh venv.
2. Public LogHub prefixes (Hugging Face):

```bash
python scripts/multi_seed_experiment.py
python scripts/bgl_experiment.py
python scripts/graph_transition_experiment.py
python scripts/if_percentile_hdfs_bgl.py
python scripts/bgl_grouping_experiment.py
```

3. HUFLIT-Career (anonymized `access.csv` is **not** released). Point the loader at a local CSV:

```bash
set HUFLIT_CAREER_CSV=data\careerhub_20260604_095930\access.csv
python scripts/huflit_baselines_nopad.py
python scripts/huflit_graph_ablation.py
```

4. Build the PDF twice: `pdflatex csonet2026.tex`.
5. Do **not** use `scripts/huflit_experiment.py` for headline HUFLIT deep numbers; it matches `logs/legacy_contamination/huflit_results.json` only.

Institutional raw HTTP logs are not redistributed.
