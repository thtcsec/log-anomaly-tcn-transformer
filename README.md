# Application of TCN and Transformer Networks for Large-Scale System Log Anomaly Detection

This repository contains the implementation, experimental source code, and academic evaluation for the paper: **"Application of TCN and Transformer Networks for Large-Scale System Log Anomaly Detection"**.

---

## 1. Overview
Log anomaly detection is a critical task in automated system operations (AIOps). This project implements a modular log anomaly detection pipeline consisting of:
1. **Log Standardization & Parsing:** Structuring raw, unstructured logs into event templates using **Drain3**.
2. **Event Grouping:** Sequencing events based on operational sessions (Block ID for HDFS) or time windows (Fixed/Sliding Windows for BGL).
3. **Anomaly Scoring:** Evaluating and comparing performance across five scoring branches:
   - **Temporal Convolutional Network (TCN):** Learning sequential patterns through Dilated Causal 1D Convolutions.
   - **Transformer:** Learning bidirectional context via Masked Event Modeling (LogBERT-inspired).
   - **Lightweight Baselines:** PCA, TruncatedSVD, Isolation Forest, and DeepLog (LSTM next-event prediction).

---

## 2. Directory Structure
```text
├── careerhub_20260604_095930/  # Real-world Career Hub server logs
├── thuvien_20260604_094551/     # Real-world Library server logs
├── images/                      # Evaluation charts (ROC/PR curves)
├── csonet2026.tex               # Springer LNCS LaTeX source code
├── csonet2026.pdf               # Compiled paper PDF
├── multi_seed_experiment.py     # HDFS multi-seed evaluations for baselines & Transformer
├── bgl_experiment.py            # Cross-dataset evaluations on BGL
├── bgl_grouping_experiment.py   # Grouping strategy evaluations on BGL
├── deeplog_experiment.py        # Baseline DeepLog LSTM replication on HDFS
├── ablation_experiment.py       # Parameter sensitivity studies (Drain3, Mask ratio, Top-k)
├── generate_explainability.py   # Qualitative explainability analysis generator
├── generate_figures.py          # Script for plotting ROC/PR curves
└── README.md                    # Project documentation
```

---

## 3. Experimental Results
Below is the evaluation summary (mean $\pm$ standard deviation) across 5 random seeds:

### HDFS Dataset (500k lines, Block ID grouping)
| Method | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| PCA | $0.6274 \pm 0.1944$ | $0.5450 \pm 0.0055$ | $0.5711 \pm 0.0944$ |
| TruncatedSVD | $0.4298 \pm 0.2555$ | $0.5527 \pm 0.0106$ | $0.4540 \pm 0.1366$ |
| Isolation Forest | $0.1726 \pm 0.0146$ | $0.1080 \pm 0.0084$ | $0.1328 \pm 0.0103$ |
| Transformer (val-tuned) | $0.5102 \pm 0.1497$ | $0.2365 \pm 0.1156$ | $0.2950 \pm 0.0693$ |
| DeepLog (top-k) | $0.9862 \pm 0.0201$ | $0.3013 \pm 0.0167$ | $0.4613 \pm 0.0199$ |
| **TCN (top-k)** | $0.9862 \pm 0.0201$ | $0.3013 \pm 0.0167$ | $0.4613 \pm 0.0199$ |

### BGL Dataset (500k lines, Window $W=100$)
| Method | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| PCA | $0.9473 \pm 0.0113$ | $0.8976 \pm 0.0144$ | $0.9217 \pm 0.0098$ |
| TruncatedSVD | $0.9462 \pm 0.0051$ | $0.9255 \pm 0.0228$ | $0.9356 \pm 0.0117$ |
| Isolation Forest | $0.1110 \pm 0.0171$ | $0.0248 \pm 0.0045$ | $0.0404 \pm 0.0070$ |
| Transformer (val-tuned) | $0.9841 \pm 0.0017$ | $0.8975 \pm 0.0157$ | $0.9388 \pm 0.0080$ |
| DeepLog (top-k) | $0.9654 \pm 0.0083$ | $0.9991 \pm 0.0012$ | $0.9820 \pm 0.0042$ |
| **TCN (top-k)** | $\mathbf{0.9666 \pm 0.0039}$ | $\mathbf{0.9991 \pm 0.0012}$ | $\mathbf{0.9826 \pm 0.0022}$ |
| **TCN (NLL)** | $0.9412 \pm 0.0085$ | $0.9537 \pm 0.0094$ | $\mathbf{0.9474 \pm 0.0035}$ |

---

## 4. Run Instructions

### Install Dependencies:
```bash
pip install torch numpy pandas scikit-learn datasets drain3 tqdm matplotlib
```

### Run Multi-seed Evaluations on HDFS:
```bash
python multi_seed_experiment.py
```

### Run Cross-dataset Evaluations on BGL:
```bash
python bgl_experiment.py
```

### Run Grouping Configuration Study:
```bash
python bgl_grouping_experiment.py
```

### Run Parameter Sensitivity Studies:
```bash
python ablation_experiment.py
```

---

## 5. Contact Info
* **First Author:** Tien-Thanh Cao (Industrial University of Ho Chi Minh City / HUFLIT) - thanhct25471@pgr.iuh.edu.vn
* **Second & Corresponding Author:** Hoang-Tu Trinh (HUFLIT) - tht.csec2005@gmail.com
* **Third Author:** Manh-Ha Tran (HUFLIT) - hatm@huflit.edu.vn
