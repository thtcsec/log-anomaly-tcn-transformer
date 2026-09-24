<p align="center">
  <img src="assets/hcmut.png" height="64" alt="HCMUT — CSoNet 2026 Organizer" />
</p>

<h1 align="center">CSoNet 2026 — Network-Aware Event Sequence Modeling for Log Anomaly Detection</h1>

<p align="center">
  <a href="https://csonet-conf.github.io/csonet26/"><img src="https://img.shields.io/badge/Conference-CSoNet%202026-0B5FFF.svg" alt="Conference" /></a>
  <a href="https://csonet-conf.github.io/csonet26/"><img src="https://img.shields.io/badge/Organizer-HCMUT%20%7C%20Bach%20Khoa-CC0000.svg" alt="Organizer" /></a>
  <a href="https://link.springer.com/conference/csonet"><img src="https://img.shields.io/badge/Proceedings-Springer%20LNCS-FF6600.svg" alt="Proceedings" /></a>
  <img src="https://img.shields.io/badge/Indexing-ISI%20%7C%20EI%20%7C%20Scopus-6A0DAD.svg" alt="Indexing" />
  <img src="https://img.shields.io/badge/Status-Camera--ready%20revision-orange.svg" alt="Status" />
</p>

---

## Paper metadata

| Field | Info |
|---|---|
| **Title** | *Network-Aware Event Sequence Modeling for User-Behavior and System Log Anomaly Detection* |
| **Venue** | CSoNet 2026 — Ho Chi Minh City, 16–18 Nov 2026 (organized at **HCMUT**) |
| **Authors** | Thanh Tien Cao, Tu Hoang Trinh, Ha Manh Tran |
| **Affiliation** | IUH & HUFLIT |
| **LaTeX** | `csonet2026.tex` (Springer LNCS) |
| **Code** | https://github.com/thtcsec/log-anomaly-tcn-transformer |

---

## Relation to prior manuscript (disclosure)

A related HDFS/BGL Drain3–TCN–Transformer manuscript was **accepted at VNICT 2026 but subsequently withdrawn before publication**:

> H.-T. Trinh and T.-T. Cao, *Application of TCN and Transformer Networks for Log Anomaly Detection in Large-Scale Enterprise and Industrial Networks*, unpublished manuscript (accepted at VNICT 2026; withdrawn before publication).

**This CSoNet manuscript extends that work** with HUFLIT-Career, explicit event-transition graph (GraphWalk) scoring, grouping/latency/explainability analyses, and proxy-label limits. **Email CSoNet TPC chairs** about the overlap (and the withdrawal) before camera-ready if not already on record.

**Prefix sizes (artifact-matched):** HDFS classical 500k / deep+GraphWalk 200k; BGL classical+deep 500k / GraphWalk+grouping 200k. See `logs/`.

**Legacy dumps:** `logs/legacy_contamination/` holds pre-percentile IF / older HUFLIT JSON. Headline IF/GraphWalk: `if_percentile_hdfs_bgl.json`, `huflit_baselines_nopad.json`, `graph_transition_results.json`.

---

## Overview

1. **Parsing** — Drain3 templates as graph vertices  
2. **Grouping** — Block ID / time window / client-IP walks  
3. **Scoring** — PCA, TruncatedSVD, Isolation Forest, DeepLog, TCN, Transformer, GraphWalk  

---

## Repository layout

```text
├── csonet2026.tex / llncs.cls     # LNCS manuscript
├── assets/hcmut.png               # Conference organizer (HCMUT)
├── images/                        # Figures for the paper
├── scripts/                       # Experiment runners (public)
├── logs/                          # Public metric dumps (incl. GraphWalk HDFS/BGL/HUFLIT)
├── data/                          # Private logs (gitignored; not published)
├── results/                       # Scratch / local reruns (gitignored)
└── README.md
```

Key public artifacts under `logs/`: `graph_transition_results.json` (GraphWalk 5-seed HDFS/BGL/HUFLIT), `deeplog_results.json`, `huflit_*.json`, grouping/latency dumps. Re-run scripts under `scripts/` to regenerate.

---

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install torch numpy pandas scikit-learn datasets drain3 tqdm matplotlib
```

Public benchmarks (HuggingFace LogHub):

```bash
python scripts/multi_seed_experiment.py
python scripts/bgl_experiment.py
python scripts/graph_transition_experiment.py   # HDFS/BGL/HUFLIT GraphWalk → logs/graph_transition_results.json
python scripts/if_percentile_hdfs_bgl.py        # unified IF 95th-pct. protocol
python scripts/bgl_grouping_experiment.py
```

HUFLIT-Career (place anonymized export under `data/careerhub_…`, never commit):

```bash
set HUFLIT_CAREER_DIR=data\careerhub_20260604_095930
python scripts/graph_transition_huflit_experiment.py
python scripts/huflit_graph_ablation.py
```

Do **not** extract the multi-GB campus RAR on `D:\huflit_logs`.

---

## Contact

- **Tu Hoang Trinh** — `tht.csec2005@gmail.com` / `23dh113972@st.huflit.edu.vn`
- **Tien Thanh Cao** — `thanhct@huflit.edu.vn`
- **Ha Manh Tran** — `hatm@huflit.edu.vn`

Faculty of Information Technology, HUFLIT · Ho Chi Minh City, Vietnam
