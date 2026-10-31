# FedEdge-GAT-IDS — Framework Phát hiện Xâm nhập RPL-IoT

Framework hoàn chỉnh cho luận văn: **Edge-aware Graph Attention Network + Federated Learning** để phát hiện xâm nhập trong mạng RPL-IoT. Hỗ trợ train / validation / test, chạy được cả centralized lẫn federated qua một cờ cấu hình, có sẵn synthetic data để chạy ngay và khung cắm dataset thật (UOS_IOTSH_2024, ROUT-4-2023, IoT-RPL).

---

## 1. Cấu trúc thư mục

```
fededge_gat_ids/
├── configs/default.yaml         # MỌI tham số tập trung ở đây
├── main.py                      # điểm vào: train/test/xai
├── requirements.txt
├── scripts/
│   └── run_ablation.py          # so sánh edge vs no-edge vs federated (KC-2)
├── src/
│   ├── data/
│   │   ├── synthetic.py         # sinh cluster-graph có chữ ký tấn công
│   │   └── loader.py            # loader linh hoạt + chia tập + chuẩn hóa
│   ├── models/
│   │   └── edge_gat.py          # Edge-aware GAT (GATv2Conv + edge_dim)
│   ├── federated/
│   │   └── fed.py               # partition non-IID + FedAvg/FedProx
│   ├── explain/
│   │   └── xai.py               # attention + GNNExplainer
│   └── utils/
│       ├── common.py            # config, seed, device, early-stopping
│       ├── metrics.py           # F1 per-class, macro-F1, confusion matrix
│       └── engine.py            # vòng train/eval dùng chung
└── tests/
```

---

## 2. Cài đặt

```bash
cd fededge_gat_ids
python -m venv .venv && source .venv/bin/activate    # khuyến nghị
pip install -r requirements.txt
```

Nếu cài `torch-geometric` gặp lỗi, cài torch trước rồi mới cài PyG theo hướng dẫn chính thức của PyG cho đúng phiên bản CUDA.

---

## 3. Chạy nhanh (3 lệnh đầu tiên)

```bash
# (1) Centralized + edge features (mặc định) — chạy ngay với synthetic
python main.py --config configs/default.yaml

# (2) Federated (bật FL)
python main.py --config configs/default.yaml --federated

# (3) Ablation: tắt edge features để so sánh
python main.py --config configs/default.yaml --no-edge
```

Mỗi lần chạy in báo cáo test (accuracy, macro-F1, F1 từng lớp) và lưu `results.json`.

Kết quả tham khảo trên synthetic (CPU, ~1 phút mỗi cấu hình):

| Cấu hình | Macro-F1 | Accuracy |
|---|---|---|
| A — node-only, centralized | ~0.90 | ~0.97 |
| B — edge-aware, centralized | ~0.90 | ~0.97 |
| C — edge-aware, **federated** | ~0.92 | ~0.98 |

> Trên synthetic, chênh lệch edge vs no-edge nhỏ vì chữ ký tấn công được đặt phần lớn ở node features. Trên **dữ liệu thật**, blackhole/selective-forwarding có tín hiệu mạnh trên cạnh → kỳ vọng edge features cải thiện rõ hơn. Đó là lý do KC-2 là một kiểm tra thực sự.

### Kết quả tối ưu trên dữ liệu UOS_IOTSH_2024 thật

Sau khi áp dụng **Dynamic Tanh (DyT, CVPR 2025)** + **Focal Loss** (gamma=1, không kèm class weight) + **residual connection**:

| Cấu hình | Accuracy | Macro-F1 | Sinkhole F1 |
|---|---|---|---|
| Ban đầu (Weighted CE + BatchNorm) | 0.970 | 0.923 | 0.863 |
| Focal nhưng dùng BatchNorm | 0.983 | 0.953 | 0.915 |
| **DyT + Focal (cấu hình mặc định)** | **0.987** | **0.965** | **0.936** |

Bài học tối ưu quan trọng:
- **KHÔNG dùng Focal Loss kèm class weight cùng lúc** — gây over-correction, precision lớp thiểu số tụt thê thảm (0.53). Chọn MỘT trong hai.
- **gamma=1** tốt hơn gamma=2 cho mức mất cân bằng này (~7% Sinkhole).
- **DyT** ổn định hơn BatchNorm với batch nhỏ / federated non-IID, +2% Sinkhole F1.
- **Làm giàu edge features bằng RPL counter** (RDAO/RDIO/SDIO/SDAO theo cặp nút): node features d_n=13, edge features d_e=10. Đây là điểm then chốt — edge features thống kê traffic thuần (pkt_count, len) gần như vô dụng (B−A=+0.002), nhưng khi nhúng counter RPL thì giá trị edge tăng gấp ~4 lần (+0.009 Sinkhole F1). Lý do: nút sinkhole hút traffic → RDAO/SDIO bất thường trên cạnh của nó.
- **Federated**: FedProx + dirichlet_alpha=1.5 (giảm non-IID) ổn định hơn FedAvg+alpha=0.5 nhiều.
- Vấn đề thật không phải accuracy (vốn đã cao do Normal chiếm đa số) mà là **F1 lớp thiểu số** — luôn nhìn macro-F1 và per-class F1, đừng chỉ nhìn accuracy.

> CẢNH BÁO về số liệu: bảng trên chạy trên một phần dataset (vài file, trộn graph rồi split ngẫu nhiên). Khi chạy đủ ~76 file và split NGẪU NHIÊN, macro-F1 thực tế khoảng 0.88–0.90. Để báo cáo trung thực trong luận văn, nên (1) split theo scenario (train/test khác file), (2) chạy nhiều seed báo cáo mean±std. Con số trên một phần dữ liệu thường lạc quan hơn thực tế.

---

## 4. Quy trình train / validation / test (giải thích cơ chế)

Đây là phần cốt lõi bạn cần nắm để trả lời hội đồng.

### 4.1. Chia tập (chống rò rỉ dữ liệu)
- Chia ở **mức graph**, không phải mức node — để không có nút nào của cùng một graph vừa nằm train vừa nằm test.
- **Stratified** theo nhãn trội của graph → giữ tỷ lệ lớp cân bằng giữa 3 tập.
- Tỷ lệ mặc định 70/15/15 (`data.split`).

### 4.2. Chuẩn hóa (fit chỉ trên train)
- `GraphScaler` tính mean/std **chỉ trên tập train**, rồi áp dụng cho val/test.
- Đây là điểm hội đồng hay hỏi: chuẩn hóa trên toàn bộ dữ liệu = data leakage. Code đã làm đúng.

### 4.3. Vòng huấn luyện
- **Train**: mỗi epoch chạy qua train_loader, tính Weighted CrossEntropy (trọng số nghịch tần suất để xử lý lớp hiếm như Sinkhole/Blackhole), backprop.
- **Validation**: sau mỗi epoch, đo macro-F1 trên val. Dùng để (a) early stopping, (b) chọn checkpoint tốt nhất.
- **Test**: chỉ chạy MỘT lần cuối, trên model tốt nhất theo val. Không bao giờ dùng test để chọn model.

### 4.4. Early stopping & checkpoint
- `EarlyStopping(patience=12, mode=max)` theo val macro-F1.
- Checkpoint tốt nhất lưu ở `checkpoints/best_centralized.pt`.

---

## 5. Chế độ Federated

Bật bằng `--federated` hoặc `federated.enabled: true`.

- **Phân vùng client**: `partition: dirichlet` tạo non-IID (mỗi client lệch về vài lớp); `alpha` nhỏ = lệch mạnh. Đổi `iid` để so sánh.
- **Vòng FL**: mỗi round = client train cục bộ `local_epochs` → server FedAvg → đánh giá val. Lặp `rounds` lần.
- **FedProx**: đổi `aggregator: fedprox` + chỉnh `fedprox_mu` nếu non-IID khó hội tụ (đúng phương án giảm thiểu rủi ro trong đề cương).
- **Client = edge gateway**, không phải sensor node (xem đề cương mục 4.x).

---

## 6. Ablation tự động (cho Chương V luận văn)

```bash
python scripts/run_ablation.py
```

Chạy 3 cấu hình A/B/C trên **cùng một split**, in bảng so sánh macro-F1, và tự kiểm tra KC-2 (edge features có cải thiện ≥ +0.02 không). Kết quả lưu `ablation_results.json` — đưa thẳng vào bảng so sánh trong luận văn.

---

## 7. Cắm dataset thật (UOS_IOTSH_2024 / ROUT-4 / IoT-RPL)

Hiện `load_from_csv()` trong `src/data/loader.py` là **khung** (trả về rỗng → fallback synthetic). Để dùng dữ liệu thật:

1. Tải dataset, đặt các file `.csv` vào `data/raw/`.
2. Mở `src/data/loader.py`, hoàn thiện `load_from_csv()`:
   - Đọc CSV bằng pandas.
   - Với mỗi (cửa sổ thời gian × cluster), gom các nút thành một graph.
   - **Node features**: map các cột RPL (rank, ETX, PDR, RSSI, đếm DIO/DAO/DIS, version, parent_change, sibling) vào `x` (shape `[n, d_n]`).
   - **Edge features**: từ quan hệ parent-child (cột `parent_id` nếu có), dựng `edge_index` và `edge_attr = [ETX_link, PDR_link, RSSI_link, hop, forward_ratio]`.
   - **Nhãn**: map cột nhãn về chỉ số lớp theo `data.classes`.
   - Trả về `list[torch_geometric.data.Data]`.
3. Đổi `data.source: csv` trong config.
4. Vì hai dataset có schema khác nhau, chuẩn hóa taxonomy nhãn về tập chung `{Normal, Sinkhole, Blackhole, Flooding, Version, Rank}`. Theo dõi **domain shift** (KC-4): nếu ghép hai nguồn làm F1 sụt > 10%, cân nhắc train riêng từng dataset trước.

> Mẹo: chạy synthetic trước để chắc pipeline đúng, rồi mới cắm dữ liệu thật — như vậy nếu lỗi, bạn biết là do dữ liệu chứ không phải do code.

---

## 8. Lộ trình theo Sprint (khớp đề cương)

| Sprint | Việc với framework này |
|---|---|
| S1 | Hoàn thiện `load_from_csv()`, kiểm tra EDA, chạy synthetic để verify pipeline |
| S2 | Train centralized (cờ mặc định), chạy `run_ablation.py` → kiểm KC-2 |
| S3 | Bật `--federated`, thử `dirichlet` vs `iid`, fedavg vs fedprox → kiểm KC-3 |
| S4 | Bật XAI, đánh giá attention + GNNExplainer → kiểm Fidelity |
| S5 | Tổng hợp `ablation_results.json` + bảng so sánh vào luận văn |

---

## 9. Mở rộng

- **Thêm baseline**: tạo model mới trong `src/models/`, giữ nguyên interface `forward(x, edge_index, edge_attr)`.
- **Đổi sang Flower**: thay `src/federated/fed.py` bằng client/server Flower; phần model/data giữ nguyên.
- **Differential Privacy**: thêm Gaussian noise vào model update trước khi gửi server.
- **GNNExplainer**: đã tích hợp; nếu phiên bản PyG khác API, xem `src/explain/xai.py` (đã bọc try/except).

---

## 10. Lưu ý quan trọng

- Con số tham số mô hình (~74K trên synthetic d_n=10) sẽ đổi theo `d_n`, `d_e`, `hidden_dim`, `heads` thật. Cập nhật lại trong luận văn sau khi chốt dữ liệu.
- Mọi seed cố định (`seed: 42`) để thí nghiệm tái lập. Đổi seed và chạy nhiều lần để báo cáo mean ± std — hội đồng đánh giá cao điều này.
