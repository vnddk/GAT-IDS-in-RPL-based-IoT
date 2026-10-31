# Kỹ thuật cải thiện F1 — có dẫn chứng khoa học

## Chẩn đoán: điểm nghẽn ở đâu?

macro-F1 EdgeGAT = 0.7782 (XGBoost 0.8052, cách 0.0271).
Tính độ nhạy: nếu nâng từng lớp lên 0.80 thì macro-F1 tăng bao nhiêu?

| Lớp | F1 hiện tại | Nếu lên 0.80 → macro-F1 | Mức tăng |
|---|---|---|---|
| **delayed_reply** | **0.042** | **0.8255** | **+0.0474** |
| rank | 0.480 | 0.7981 | +0.0200 |
| clone_id | 0.618 | 0.7895 | +0.0114 |
| continuous_sinkhole | 0.641 | 0.7881 | +0.0099 |
| sybil | 0.680 | 0.7856 | +0.0075 |

→ **delayed_reply một mình kéo −0.047**. Đây là mục tiêu số 1.
→ Nhóm "họ-rank" (rank/sinkhole/cont_sinkhole/clone_id/sybil) đều 0.48–0.79,
  dấu hiệu chúng NHẦM LẪN CHÉO với nhau (cùng thao túng rank).

---

## Kỹ thuật 1 — Đặc trưng ngữ cảnh thời gian (multi-window)

**Nguồn:**
- *Temporal Analysis Framework for Intrusion Detection Systems* (arXiv:2511.03799,
  2025): khảo sát hơn 40 công trình 2020–2025, kết luận nhóm phương pháp
  **inter-flow sequential và temporal window-based có độ phủ thời gian rộng nhất**
  trên chuỗi MITRE ATT&CK.
- *PPT-GNN* (arXiv:2406.13365): học trên **temporal sliding window snapshots**,
  gộp thông tin liên- và nội-snapshot; nêu rõ lấy mẫu ngẫu nhiên phá vỡ cấu trúc
  thời gian và gây rò rỉ train→test.
- *DMSTG-AD* (Scientific Reports, 2026): GRU-driven dynamic node embeddings +
  bidirectional GRU cho phụ thuộc thời gian.
- *GCN-2-Former* (Scientific Reports, 2025): sliding window + dynamic graph.

**Vì sao hợp với ta:** framework đang xử lý mỗi cửa sổ 5 giây ĐỘC LẬP.
`delayed_reply` = "node trả lời chậm hơn BÌNH THƯỜNG" — chỉ lộ khi so node với
CHÍNH NÓ trước đó, vì mỗi node có baseline trễ khác nhau.

**Kiểm chứng (mô phỏng, Cohen's d):**

| Cách đo | Cohen's d |
|---|---|
| resp_delay tuyệt đối, 1 cửa sổ (hiện tại) | 1.81 |
| tỉ lệ so với lịch sử chính node (mới) | **7.40** |

**Đã hiện thực:** 4 đặc trưng mới (node 21 → 25), chỉ dùng QUÁ KHỨ (nhân quả,
không rò rỉ), EWMA α=0.3:

```
#21 resp_delay_ratio = log1p( resp_delay(t) / EWMA_hist(resp_delay) )
#22 iat_ratio        = log1p( iat_std(t)    / EWMA_hist(iat_std)    )
#23 fwd_ratio_delta  = fwd_ratio(t)  − EWMA_hist(fwd_ratio)
#24 rank_delta       = rank_mean(t)  − EWMA_hist(rank_mean)
```

`log1p` là bắt buộc: tỉ lệ thô bùng nổ khi mẫu số ~0 → phương sai khổng lồ.
Đo trên builder thật: d = 0.30 (tỉ lệ thô) → **1.45** (sau log1p).

---

## Kỹ thuật 2 — Label smoothing

**Nguồn:** *A Study on a NIDS Based on the Fusion of SAGEConv-GNN and a
Transformer Encoder* (MDPI Electronics 15(8):1737, 2026) — dùng **class-weighted
cross-entropy KÈM label smoothing** để vừa xử lý mất cân bằng vừa cải thiện tổng
quát hoá; kèm gradient clipping và early stopping theo macro-F1 (ta đã có sẵn 2
thứ sau).

**Vì sao hợp với ta:** 5 lớp họ-rank có chữ ký gần giống nhau. Nhãn cứng one-hot
ép mô hình cực tự tin vào một lớp, khuếch đại lỗi khi ranh giới mờ. Label
smoothing làm mềm mục tiêu, giảm quá tự tin.

**Đã hiện thực:** `ClassBalancedFocalLoss(label_smoothing=0.05)`.
Bật/tắt trong `configs/default.yaml` → `train.label_smoothing`.

---

## Kỹ thuật ĐÃ CÂN NHẮC nhưng CHƯA áp dụng

| Kỹ thuật | Nguồn | Vì sao hoãn |
|---|---|---|
| GAN + Soft Nearest Neighbor Loss | Future Internet 17(5):216, 2025 | Phức tạp; phải huấn luyện GAN riêng. Lợi ích trên NSL-KDD chỉ ~+0.02 F1 lớp hiếm |
| ADASYN thay SMOTE | ICIoTML 2024 | Ta đã có GraphSMOTE-lite bám cấu trúc đồ thị; đổi sang ADASYN cần viết lại phần chèn node |
| Hợp nhất GNN + Transformer Encoder | MDPI Electronics 2026 | Thay đổi kiến trúc lớn, rủi ro cao. Nên thử SAU khi 2 kỹ thuật trên đã đo xong |
| Chọn đặc trưng bằng GNN | CMC 2025 | Ta đã có kiểm toán + xếp hạng lai |

---

## Kỳ vọng thành thật

- Kỹ thuật 1 nhắm thẳng `delayed_reply`. Nếu nó lên ~0.6–0.8, macro-F1 tăng
  khoảng **+0.03 đến +0.05**.
- Kỹ thuật 2 thường cho mức tăng **khiêm tốn (+0.005 đến +0.02)**, chủ yếu ở
  các lớp dễ nhầm.
- Không có gì bảo đảm. Cohen's d đo trên mô phỏng, dữ liệu thật có thể khác.
  Nếu `delayed_reply` vẫn thấp sau khi chạy, gửi log để phân tích tiếp.

## Chạy

```bash
python scripts/convert_data.py --dataset radar --data-dir data/radar --window 5
python scripts/train_all.py --epochs 120
python scripts/test_all.py
```

Log convert phải thấy `d_n=25 d_e=10` và kiểm toán vẫn báo "Không phát hiện rò rỉ".
