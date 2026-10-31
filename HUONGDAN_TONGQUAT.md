# Hướng dẫn tổng quát — GAT-IDS Framework

Tài liệu này là bản đồ của toàn bộ khung phần mềm. Đọc `README.md` trước để
nắm ý tưởng, rồi dùng tài liệu này để biết chạy cái gì ở đâu.

---

## 1. Bốn mô hình đề xuất

| Mô hình | Tệp | Tham số (RADAR) | Bài toán giải |
|:---|:---|---:|:---|
| **GAT-IDS** | `src/models/edge_gat.py` | 103.956 | Phân loại nút đa lớp, có giám sát |
| **NE-GAT-IDS** | `src/models/ne_gat.py` | 62.224 | Như trên, kiến trúc hai kênh |
| **GAT-Transformer IDS** | `src/models/gat_transformer.py` | ~236.000 | Tự giám sát + tinh chỉnh; phát hiện tấn công chưa từng thấy |
| **T-GAT-IDS** | `src/models/temporal_gat.py` | ~100.000 | Mở rộng động theo thời gian giữa các cửa sổ |

Sáu mô hình đối chứng: RF, XGBoost, LightGBM (`tabular_baselines.py`);
MLP, GCN, E-GraphSAGE (`gnn_baselines.py`).

---

## 2. Ba bước bắt buộc

```bash
# Bước 1 — đồ thị hoá, chia theo THỜI GIAN, kiểm toán rò rỉ
python scripts/convert_data.py --dataset radar --data-dir data/radar \
       --split chrono --alpha 1.0 --out-dir data/radar_chrono

# Bước 2 — huấn luyện
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 3 --lambda-d 0.5 --disc mmd --epochs 120 --ckpt-dir ck_negat

# Bước 3 — đánh giá đa lớp VÀ nhị phân
python scripts/test_all.py --data-dir data/radar_chrono \
       --ckpt-dir ck_negat --binary
```

**Sau Bước 1 phải kiểm hai dòng log:**

- `Chẩn đoán rò rỉ thời gian: δ̄ = …` — phải lớn hơn 2 rõ rệt.
- `Không phát hiện rò rỉ` — nếu có `WARNING — Đặc trưng RÒ RỈ` thì dừng lại xử lý.

---

## 3. Bảng tra script

| Việc cần làm | Lệnh |
|:---|:---|
| Đồ thị hoá + chia tập | `convert_data.py` |
| Huấn luyện có/không ATA, mọi kiến trúc | `train_ata.py --arch {edgegat,negat,mlp,gcn,egraphsage}` |
| Huấn luyện mô hình cây | `train_all.py --models RF,XGBoost,LightGBM` |
| Tự giám sát + tinh chỉnh | `train_ssl.py --stage {ssl,finetune,scratch}` |
| Đánh giá | `test_all.py --binary` |
| So sánh không gian đặc trưng cuối | `tsne_compare.py` |
| Trực quan hoá theo tầng | `tsne_layers.py` |
| Ma trận nhầm lẫn | `confusion_matrix.py` |
| Khả diễn giải | `attention_per_class.py`, `feature_importance_per_class.py`, `xai_simple.py` |
| Gộp nhiều hạt giống | `aggregate_seeds.py` |

---

## 4. Các cờ quan trọng của `train_ata.py`

| Cờ | Ý nghĩa |
|:---|:---|
| `--arch` | Kiến trúc: `edgegat`, `negat`, `mlp`, `gcn`, `egraphsage` |
| `--K`, `--lambda-d` | Số miền thời gian và trọng số căn chỉnh. `--K 1 --lambda-d 0` = **nhánh đối chứng không ATA** |
| `--disc` | Độ đo chênh lệch: `coral`, `mmd`, `cosine` |
| `--domain-strategy` | `quantile` hoặc `tdc` |
| `--no-edge` | **TẮT đặc trưng cạnh** trong hệ số chú ý (ablation) |
| `--edge-score` | NE-GAT: `full` hoặc `edge_only` (phép thử giả thuyết lấn át) |
| `--edge-agg` | NE-GAT: `mean_deg` hoặc `mean` (đo tác động của thành phần khôi phục bậc) |
| `--hidden-dim`, `--out-dim`, `--clf-hidden`, `--num-layers` | Nâng dung lượng để **so sánh công bằng** |
| `--cls-reduction` | `mean` (mặc định) hoặc `sum` (tái lập đúng công thức gốc) |

---

## 5. Nâng dung lượng baseline cho công bằng

Chiều ẩn đã dò sẵn để đạt tương đương tham số với hai mô hình đề xuất:

| Baseline | ≈ 62k (bằng NE-GAT) | ≈ 104k (bằng GAT-IDS) |
|:---|:---|:---|
| E-GraphSAGE | `--hidden-dim 184` → 60.882 | `--hidden-dim 256` → 102.354 |
| MLP | `--hidden-dim 216` → 61.755 | `--hidden-dim 288` → 102.867 |
| GCN | `--hidden-dim 960` → 62.066 | không đạt được |

GCN cần chiều ẩn 960 mới đạt 62k vì kiến trúc quá gọn; ở mức đó gần như
chắc chắn quá khớp. Nên báo cáo GCN ở cả hai mức và nêu rõ nó không scale
được — bản thân điều đó là một quan sát có giá trị.

---

## 6. Ba cạm bẫy đã gặp và cách tránh

**Cạm bẫy 1 — cờ bị argparse nuốt im lặng.** Trước đây `--no-edge` bị hiểu
thành `--no-edge-update` do khớp tiền tố, khiến một thí nghiệm ablation cho
kết quả trùng khít với nhánh đối chứng mà không báo lỗi nào. Đã đặt
`allow_abbrev=False`. **Luôn kiểm số tham số trong log**: nhánh `--no-edge`
phải có ít hơn đúng 3.840 tham số.

**Cạm bẫy 2 — nhầm đường trả đặc trưng.** Mô hình có hai đường: đường phục
vụ trực quan hoá gọi `detach()`, đường phục vụ ATA thì không. Nhầm hai
đường sẽ khiến số hạng căn chỉnh **không có tác dụng gì** mà không báo lỗi.

**Cạm bẫy 3 — đơn vị của δ̄ khác nhau giữa các bộ.** RADAR và UOS đếm theo
cửa sổ 5 giây, RPL-Beh theo chỉ số phút. Quy về giây trước khi so sánh:
302 s, 75 s và 7.440 s. Ngoài ra δ̄ ≈ 0,2·n nên nó là chỉ số **tương đối**
theo độ dài tệp, không phải ngưỡng tuyệt đối.

---

## 7. Việc bắt buộc trước khi nộp

1. **Năm hạt giống** cho mọi cấu hình chính, báo cáo trung bình kèm độ lệch chuẩn.
2. **Nhánh đối chứng** `--K 1 --lambda-d 0` cho mỗi kiến trúc.
3. **Chạy lại ablation `--no-edge`** bằng mã đã vá.
4. **Kiểm số tham số** trong mọi log trước khi tin một kết quả ablation.

---

## 8. Cạm bẫy 4 — nút thắt của bộ tự mã hoá (QUAN TRỌNG)

Trong `GAT-Transformer IDS`, tham số `d_model` **phải nhỏ hơn hoặc bằng**
chiều embedding của GNN (`out_dim`, mặc định 32).

Nếu `d_model > out_dim` thì bộ tự mã hoá **không có nút thắt nào**, nên nó
học được hàm **đồng nhất** — tái tạo hoàn hảo MỌI đầu vào, kể cả nút tấn
công. Khi đó sai số tái tạo tiến về 0 với tất cả mẫu và điểm bất thường mất
sạch khả năng phân biệt.

**Dấu hiệu trong nhật ký:** `MSE` tụt về `0.00000` và `ROC-AUC` rơi xuống
quanh hoặc **dưới** 0,5.

Đo được trên tập synthetic, cùng mọi thứ khác:

| Cấu hình | Tham số | ROC-AUC | PR-AUC | F1 tấn công | Bỏ lọt |
|:---|---:|---:|---:|---:|---:|
| `d_model=64` (sai, không nút thắt) | 237.936 | 0,7685 | 0,6024 | 0,2464 | 85,8 % |
| `d_model=32` (đúng) | **109.502** | **0,9365** | **0,8589** | **0,7708** | **19,9 %** |

Mặc định đã đổi thành `d_model=32`, `dim_ff=64`. Mô hình cảnh báo khi phát
hiện `d_model > d_gnn`.

**Bảng tra dung lượng:**

| Cấu hình | Tổng | Khối MAE | Giảm |
|:---|---:|---:|---:|
| `--d-model 64 --dim-ff 128 --tr-layers 2` | 237.936 | 171.616 | — |
| `--d-model 32 --dim-ff 64 --tr-layers 2` *(mặc định)* | 111.184 | 44.864 | 53 % |
| `--d-model 32 --dim-ff 64 --tr-layers 1` | 89.808 | 23.488 | 62 % |
| `--d-model 24 --dim-ff 48 --tr-layers 1` | 80.104 | 13.784 | 66 % |
