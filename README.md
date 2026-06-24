# Ứng dụng mạng TCN và Transformer trong phát hiện bất thường từ dữ liệu log hệ thống quy mô lớn

Dự án này chứa mã nguồn thực nghiệm và tài liệu học thuật của nghiên cứu ứng dụng mạng tích chập thời gian (**TCN**) và mô hình tự chú ý (**Transformer**) để phát hiện bất thường từ chuỗi sự kiện log hệ thống quy mô lớn, nộp tham dự Hội thảo Quốc gia VNICT 2026.

---

## 1. Tổng quan nghiên cứu
Phát hiện bất thường từ dữ liệu log hệ thống là một bài toán quan trọng trong vận hành mạng doanh nghiệp (AIOps). Dự án xây dựng một pipeline chuẩn mực bao gồm:
1. **Log Standardisation & Parsing:** Sử dụng **Drain3** để cấu trúc hóa log thô thành các mã sự kiện cố định (Event ID).
2. **Event Grouping:** Gom chuỗi theo cấu trúc vận hành (Block ID đối với HDFS) hoặc cửa sổ thời gian cố định (Fixed Window đối với BGL).
3. **Anomaly Scoring:** So sánh hiệu năng của hai nhánh chính:
   * **Temporal Convolutional Network (TCN):** Học chuỗi thông qua các lớp tích chập nhân quả giãn nở (Dilated Causal 1D Convolution).
   * **Transformer (LogBERT-inspired):** Học ngữ cảnh chuỗi hai chiều bằng tác vụ dự đoán token bị che (Masked Event Modeling).
   * **Baselines so sánh:** PCA, TruncatedSVD, Isolation Forest, và DeepLog (LSTM Next-Event Prediction).

---

## 2. Cấu trúc thư mục dự án
```text
├── archive/                  # Lưu trữ các mẫu định dạng zip và file cũ
├── IEEECS_CPS_2026/          # Thư mục chứa template LaTeX chuẩn IEEE Computer Society
│   └── IEEEtran.cls          # Định dạng class chính cho tài liệu LaTeX
├── images/                   # Chứa các biểu đồ kết quả (ROC, PR curves) chèn vào LaTeX
├── vnict2026.tex             # File mã nguồn LaTeX chính của bài báo
├── vnict2026.pdf             # Bản PDF bài báo sau khi biên dịch
├── multi_seed_experiment.py  # Thực nghiệm HDFS đa seed cho baseline và Transformer
├── bgl_experiment.py         # Thực nghiệm đánh giá chéo trên dataset BGL
├── deeplog_experiment.py     # Thực nghiệm tái lập DeepLog LSTM trên HDFS
├── ablation_experiment.py    # Các khảo sát độ nhạy siêu tham số (Drain3, Mask ratio, Top-k)
├── generate_figures.py       # Script vẽ biểu đồ ROC/PR và phân phối điểm bất thường
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 3. Kết quả thực nghiệm hiện tại
Dưới đây là kết quả thực nghiệm trung bình $\pm$ độ lệch chuẩn trên 5 random seed:

### Tập dữ liệu HDFS (500k log, Block ID)
| Phương pháp | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| PCA | $0.6274 \pm 0.1944$ | $0.5450 \pm 0.0055$ | $0.5711 \pm 0.0944$ |
| TruncatedSVD | $0.4298 \pm 0.2555$ | $0.5527 \pm 0.0106$ | $0.4540 \pm 0.1366$ |
| Isolation Forest | $0.1726 \pm 0.0146$ | $0.1080 \pm 0.0084$ | $0.1328 \pm 0.0103$ |
| Transformer (val-tuned) | $0.5102 \pm 0.1497$ | $0.2365 \pm 0.1156$ | $0.2950 \pm 0.0693$ |
| DeepLog (top-k) | $0.9862 \pm 0.0201$ | $0.3013 \pm 0.0167$ | $0.4613 \pm 0.0199$ |
| **TCN (top-k) - Integrated** | $0.9862 \pm 0.0201$ | $0.3013 \pm 0.0167$ | $0.4613 \pm 0.0199$ |
| **TCN (NLL) - Integrated** | $0.2621 \pm 0.0277$ | $0.3644 \pm 0.0291$ | $0.3042 \pm 0.0236$ |

### Tập dữ liệu BGL (500k log, Cửa sổ $W=100$)
| Phương pháp | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| PCA | $0.9473 \pm 0.0113$ | $0.8976 \pm 0.0144$ | $0.9217 \pm 0.0098$ |
| TruncatedSVD | $0.9462 \pm 0.0051$ | $0.9255 \pm 0.0228$ | $0.9356 \pm 0.0117$ |
| Isolation Forest | $0.1110 \pm 0.0171$ | $0.0248 \pm 0.0045$ | $0.0404 \pm 0.0070$ |
| Transformer (val-tuned) | $0.9841 \pm 0.0017$ | $0.8975 \pm 0.0157$ | $0.9388 \pm 0.0080$ |
| DeepLog (top-k) | $0.9654 \pm 0.0083$ | $0.9991 \pm 0.0012$ | $0.9820 \pm 0.0042$ |
| **TCN (top-k) - Integrated** | $0.9666 \pm 0.0039$ | $0.9991 \pm 0.0012$ | **$0.9826 \pm 0.0022$** |
| **TCN (NLL) - Integrated** | $0.9412 \pm 0.0085$ | $0.9537 \pm 0.0094$ | **$0.9474 \pm 0.0035$** |

---

## 4. Minh chứng thực nghiệm (Experiment Logs & Proofs)
Các kết quả thực nghiệm báo cáo trong bài viết được lưu trữ chi tiết dưới dạng log huấn luyện và file kết quả thô trong thư mục [logs/](logs/). Người đọc có thể đối chiếu các file này làm minh chứng chạy thực nghiệm:
* **Kết quả huấn luyện và đánh giá mô hình (DeepLog, TCN, Transformer):**
  * [logs/deeplog.log](logs/deeplog.log) & [logs/deeplog_results.json](logs/deeplog_results.json): Log huấn luyện từng epoch, hàm loss, và kết quả Precision/Recall/F1 chi tiết của mô hình DeepLog LSTM trên HDFS.
  * [logs/bgl.log](logs/bgl.log) & [logs/bgl_results.json](logs/bgl_results.json): Log huấn luyện chi tiết 5 seeds cho DeepLog, TCN, và Transformer trên bộ dữ liệu BGL.
  * [logs/multi_seed.log](logs/multi_seed.log) & [logs/multi_seed_results.json](logs/multi_seed_results.json): Log chạy 5 seeds trên HDFS của baselines và các mô hình chuỗi.
* **Ablation Study (Khảo sát độ nhạy):**
  * [logs/ablation.log](logs/ablation.log) & [logs/ablation_results.json](logs/ablation_results.json): Kết quả đo độ nhạy của ngưỡng Drain3, top-K, và tỷ lệ mask của Transformer.
* **Độ trễ suy luận (Inference Latency):**
  * [logs/latency.log](logs/latency.log) & [logs/latency_results.json](logs/latency_results.json): Kết quả benchmark thời gian chạy thực tế của TCN, LSTM, Transformer trên CPU/GPU.
* **Dữ liệu phân tích định tính (Qualitative analysis):**
  * [logs/qualitative.log](logs/qualitative.log) & [logs/qualitative_examples.json](logs/qualitative_examples.json): Các chuỗi log bình thường và bất thường cụ thể phục vụ cho phân tích trong bài viết.

---

## 5. Hướng dẫn chạy thực nghiệm

### Cài đặt thư viện:
```bash
pip install torch numpy pandas scikit-learn datasets drain3 tqdm matplotlib
```

### Chạy thực nghiệm đa seed trên HDFS:
```bash
python multi_seed_experiment.py
```

### Chạy thực nghiệm trên BGL:
```bash
python bgl_experiment.py
```

### Chạy kiểm tra độ nhạy (Ablation study):
```bash
python ablation_experiment.py
```

---

## 6. Thành viên thực hiện & Giảng viên hướng dẫn
* **Tác giả:** **Trịnh Hoàng Tú** (Khoa Công nghệ thông tin, Trường Đại học Ngoại ngữ - Tin học TP.HCM - HUFLIT)
  * Email: tht.csec2005@gmail.com
* **Giảng viên hướng dẫn:** **ThS. Cao Tiến Thành** (Khoa Công nghệ thông tin, Trường Đại học Ngoại ngữ - Tin học TP.HCM - HUFLIT)


