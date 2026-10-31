# LightGBM baseline + ba loại tầng phân loại

## 1. Baseline LightGBM

Thêm vào danh sách mô hình đối chứng, dùng CÙNG bộ đặc trưng phẳng như RF và
XGBoost (đặc trưng node + tổng hợp láng giềng).

Cấu hình: `n_estimators=300, num_leaves=63, learning_rate=0.08, subsample=0.9,
colsample_bytree=0.9, min_child_samples=10, class_weight="balanced"`.

Lý do chọn `class_weight="balanced"`: LightGBM có tham số này (XGBoost không),
nên ba mô hình cây bổ trợ nhau về góc nhìn xử lý mất cân bằng — RF và LightGBM
có cân bằng lớp, XGBoost thì không.

```bash
python scripts/train_all.py --models LightGBM
python scripts/train_all.py --models RF,XGBoost,LightGBM     # cả ba cây
python scripts/test_all.py
```

`test_all.py` tự nhận và đưa vào bảng xếp hạng chung.

---

## 2. Ba loại tầng phân loại

| Kiểu | Cấu trúc | Tham số (L=3) |
|---|---|---|
| `linear` | Linear(61 → 16) | 92.996 |
| `mlp` | Linear(61→64) → ELU → Dropout → Linear(64→16) | 97.012 |
| `lstm` | LSTM trên chuỗi tầng → Linear | 127.876 |

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3 --classifier mlp --ckpt-dir ck_mlp

python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3 --classifier lstm --ckpt-dir ck_lstm
```

Chiều ẩn chỉnh bằng `--clf-hidden` (mặc định 64).

---

## ⚠ LƯU Ý QUAN TRỌNG VỀ 'LSTM CLASSIFIER'

LSTM là mạng xử lý CHUỖI, nhưng đầu ra EdgeGAT là một VECTOR cho mỗi node
(z ∈ ℝ⁶¹), không phải chuỗi. Có hai cách hiện thực:

### Cách 1 — SAI về nguyên lý (KHÔNG dùng)
Coi 61 chiều của vector như chuỗi dài 61. Sai vì **thứ tự các chiều là tuỳ ý**:
`dis_sent` đứng trước `n_sent` chỉ vì thứ tự viết code, không có quan hệ thời
gian nào. Áp LSTM lên đó là ép mô hình học một cấu trúc KHÔNG TỒN TẠI — thêm
tham số, thêm rủi ro overfit, không thêm thông tin.

### Cách 2 — ĐÃ HIỆN THỰC, có cơ sở công bố
Coi biểu diễn CỦA TỪNG TẦNG là chuỗi: **h¹ → h² → h³**

Đây là chuỗi có thật: mỗi bước ứng với bán kính thu nhận 1-hop, 2-hop, 3-hop.
Thứ tự có ý nghĩa vật lý rõ ràng.

Chính là biến thể LSTM của **Jumping Knowledge Network** (Xu, Li, Tian, Sonobe,
Kawarabayashi, Jegelka — ICML 2018). Bài báo nêu ba cách gộp biểu diễn nhiều
tầng: nối (concat), lấy max, và LSTM-attention.

Cơ chế: LSTM đọc chuỗi 3 bước rồi quyết định mỗi node nên dựa vào bán kính nào.
Node ở rìa mạng có thể cần 3-hop; node trung tâm có thể chỉ cần 1-hop.

**Chỉ có ý nghĩa khi `num_layers ≥ 2`.** Với 1 tầng, chuỗi chỉ có 1 bước và
LSTM thoái hoá thành một phép biến đổi tuyến tính đắt tiền.

Nếu anh muốn thử Cách 1 để so sánh, nói tôi biết — nhưng tôi khuyên ghi rõ
trong luận văn rằng nó không có cơ sở, tránh bị hội đồng chất vấn.

---

## Bộ thí nghiệm đề xuất

```bash
# Ba mô hình cây (nhanh, vài phút)
python scripts/train_all.py --models RF,XGBoost,LightGBM --ckpt-dir ck_main

# Ba classifier trên cấu hình EdgeGAT tốt nhất
for c in linear mlp lstm; do
  python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
      --layer-residual --resample off --weight-cap 3 --classifier $c --ckpt-dir ck_clf_$c
  python scripts/test_all.py --ckpt-dir ck_clf_$c
done
```

Kỳ vọng thành thật: `mlp` có thể giúp nhóm lớp chồng lấn (họ-rank) nhờ ranh
giới phi tuyến. `lstm` thêm 35 nghìn tham số nên rủi ro overfit cao trên đồ thị
nhỏ — nếu không cải thiện, đó cũng là kết quả đáng báo cáo.
