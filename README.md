<p align="center">
  <img src="assets/hcmut.png" height="64" alt="HCMUT — CSoNet 2026 Organizer" />
  <br />
  <sub>Conference organizer. Authors are affiliated with IUH and HUFLIT, not HCMUT.</sub>
</p>

<h1 align="center">Network-Aware Event Sequence Modeling</h1>
<p align="center"><em>User-behavior and system log anomaly detection</em></p>

<p align="center">
  <a href="https://csonet-conf.github.io/csonet26/"><img src="https://img.shields.io/badge/CSoNet-2026-0B5FFF.svg" alt="CSoNet 2026" /></a>
  <a href="https://github.com/thtcsec/log-anomaly-tcn-transformer/tree/csonet2026-camera-ready"><img src="https://img.shields.io/badge/artifact-camera--ready-2ea44f.svg" alt="Camera-ready artifact tag" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-blue.svg" alt="MIT license" /></a>
  <a href="ARTIFACT.md"><img src="https://img.shields.io/badge/reproducibility-logs%20dumps-informational.svg" alt="Frozen metric dumps" /></a>
</p>

## Paper metadata

| Field | Record |
|---|---|
| **Title** | *Network-Aware Event Sequence Modeling for User-Behavior and System Log Anomaly Detection* |
| **Authors** | Thanh Tien Cao, Tu Hoang Trinh, and Ha Manh Tran* |
| **Affiliation** | IUH (Cao) and Faculty of Information Technology, HUFLIT |
| **Venue** | CSoNet 2026, HCMUT, Ho Chi Minh City, 16–18 November 2026 |
| **Status** | Accepted; Springer LNCS forthcoming. No DOI is claimed. |
| **Artifact tag** | [`csonet2026-camera-ready`](https://github.com/thtcsec/log-anomaly-tcn-transformer/tree/csonet2026-camera-ready) |
| **Code license** | MIT (`LICENSE`). LogHub corpora remain under their owners' terms. Institutional HTTP logs are not released. |

\*Corresponding authors: `hatm@huflit.edu.vn`, `tht.csec2005@gmail.com`

## Overview

This is an applied evaluation study, not a new detector architecture. Drain3 templates are vertices of a directed transition graph; a Block ID, time window, or client session is a walk on that graph. Walks are scored with GraphWalk (Laplace-smoothed \(P(v\mid u)\)) alongside PCA/SVD, Isolation Forest, a DeepLog-style LSTM, TCN, and a masked Transformer.

## Headline results

Cross-dataset F1 (mean±std, 5 seeds). Prefix sizes are **not** matched across every cell: HDFS classical 500k / neural+GraphWalk 200k; BGL 500k except GraphWalk 200k. HUFLIT classical/GraphWalk: client-disjoint. Do not rank historical HUFLIT deep rows against current classical/GraphWalk.

| Method | HDFS | BGL | HUFLIT |
|---|---|---|---|
| PCA | 0.5711±0.0944 | 0.9217±0.0098 | 0.4583±0.2633 |
| Isolation Forest | 0.0381±0.0174 | 0.0172±0.0063 | 0.5395±0.2982 |
| DeepLog-style (top-k) | 0.4613±0.0199 | 0.9820±0.0042 | historical dump |
| TCN (top-k) | 0.4613±0.0199 | **0.9826±0.0022** | historical dump |
| GraphWalk (NLL) | 0.3082±0.0278 | 0.2328±0.0474 | 0.4963±0.2698 |

Source: `logs/` dumps listed in `ARTIFACT.md`. HUFLIT labels are heuristic scanner-probe rules, not SOC ground truth.

## Relation to a withdrawn manuscript

A related HDFS/BGL Drain3–TCN–Transformer manuscript was accepted at VNICT 2026 and **withdrawn before publication**. This CSoNet paper extends that work with GraphWalk, HUFLIT-Career, and grouping/latency analyses. It is a different question from the SOICT 2026 campus-archive validity study.

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Public LogHub prefixes:

```bash
python scripts/multi_seed_experiment.py
python scripts/bgl_experiment.py
python scripts/graph_transition_experiment.py
python scripts/if_percentile_hdfs_bgl.py
python scripts/bgl_grouping_experiment.py
```

HUFLIT-Career (local CSV only; never commit):

```bash
set HUFLIT_CAREER_CSV=data\careerhub_20260604_095930\access.csv
python scripts/huflit_baselines_nopad.py
python scripts/huflit_graph_ablation.py
```

Do **not** use `scripts/huflit_experiment.py` for headline HUFLIT deep numbers (`logs/legacy_contamination/` only).

The compiled manuscript PDF is not published in this repository. Source: `csonet2026.tex`. PDF after SpringerLink.

## Limitations

Prefix sizes and splits differ across scorers, so HDFS classical vs deep F1 is not a head-to-head ranking. Drain3 is fit on the evaluated prefix. HUFLIT Isolation Forest F1 has large seed variance. No operational incident-detection claim.

## Citation

```bibtex
@inproceedings{cao2026networkaware,
  author    = {Thanh Tien Cao and Tu Hoang Trinh and Ha Manh Tran},
  title     = {Network-Aware Event Sequence Modeling for User-Behavior and System Log Anomaly Detection},
  booktitle = {Proc. 15th Int. Conf. Computational Social Networks (CSoNet)},
  year      = {2026},
  note      = {Accepted manuscript. Artifact: https://github.com/thtcsec/log-anomaly-tcn-transformer/tree/csonet2026-camera-ready}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff). Evaluation protocol: [`ARTIFACT.md`](ARTIFACT.md).

## License

Code is MIT. Do not redistribute LogHub dumps or HUFLIT HTTP logs from this repository.
