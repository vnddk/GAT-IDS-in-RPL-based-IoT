# v3.11 — Chuẩn hoá trọng số có trần (đã BỎ Group DRO)

## Vì sao BỎ Group DRO — anh đã đúng

Group DRO cập nhật trọng số bằng `q_c ← q_c · exp(η · L_c)`. Bước dịch chuyển phụ
thuộc TÍCH `η·L_c`, mà `L_c` **không có thang chuẩn** — đúng như anh nhận xét,
thuật toán thiếu một hằng số tham chiếu.

Tôi đã dò η = 0,01 trên MÔ PHỎNG với loss giả (trung bình 0,7). Nhưng loss THẬT
trong log của anh nhỏ hơn ~50 lần:

| Kịch bản | L trung bình | η·L | q_max sau 200 bước |
|---|---|---|---|
| mô phỏng của tôi | 0,700 | 0,0070 | 0,499 |
| thực tế epoch 1 | 0,079 | 0,00079 | 0,089 |
| **thực tế epoch 120** | **0,0129** | **0,000129** | **0,066** |

Với phân phối đồng đều q = 1/16 = 0,0625, giá trị 0,066 nghĩa là **q gần như
không dịch chuyển** — Group DRO trở thành phép toán vô nghĩa.

Tệ hơn: muốn nó hoạt động phải chỉnh η theo thang loss, mà thang loss **thay đổi
trong quá trình huấn luyện** (0,079 → 0,0129). Không có hằng số ổn định nào để
neo vào. **Đã gỡ hoàn toàn khỏi framework.**

---

## Thay bằng: chuẩn hoá trọng số lớp CÓ TRẦN

    E_c = (1 − β^{n_c}) / (1 − β)         [số mẫu hiệu dụng, Cui 2019]
    w_c = 1 / E_c
    w   = clip(w, w_min, cap · w_min)     [CHẶN TRẦN — mới]
    w   = w / mean(w)                      [chuẩn hoá trung bình = 1]

`cap` là **hằng số tham chiếu rõ ràng và ổn định**: tỉ lệ tối đa cho phép giữa
lớp được ưu tiên nhất và lớp ít nhất. Không phụ thuộc thang loss, không thay đổi
theo epoch.

### Trọng số trên phân bố thật của RADAR (β = 0,9999)

| cap | w(Normal) | w(hiếm nhất) | tỉ lệ | |
|---|---|---|---|---|
| tắt | 0,1023 | 1,8369 | **18,0** | hành vi cũ — bù quá đà |
| 10 | 0,1553 | 1,5533 | 10,0 | |
| **5 (mặc định)** | **0,2693** | **1,3464** | **5,0** | KHUYẾN NGHỊ |
| 3 | 0,3863 | 1,1589 | 3,0 | cân bằng nhất, ưu tiên accuracy |

Trung bình luôn = 1,0 nên thang loss tổng thể không đổi.

### Bằng chứng cần trần

Lần chạy 120 epoch (tỉ lệ 18,0, không trần) cho thấy mô hình **bù quá đà**:

| Lớp | Precision | Recall | Hút nhầm |
|---|---|---|---|
| delayed_reply | 0,391 | 0,978 | 408 node |
| sinkhole | 0,676 | 0,875 | 433 node |
| clone_id | 0,568 | 0,735 | 881 node |

Recall cao nhưng precision thấp — lớp hiếm đang hút node của lớp khác vào.
Đó là hệ quả trực tiếp của trọng số cách biệt 18 lần.

---

## Hai chỗ áp trần (cùng nguyên tắc)

| Nơi | Tham số | Mặc định |
|---|---|---|
| Trọng số hàm mất mát | `train.weight_cap` / `--weight-cap` | 5,0 |
| Trọng số lấy mẫu lại | `--sampler-cap` | 5,0 |

---

## Lệnh chạy

```bash
# Mốc hiện tại (không trần, macro-F1 0,8370)
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off --weight-cap 0 --ckpt-dir ck_nocap

# Có trần — khuyến nghị
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off --weight-cap 5 --ckpt-dir ck_cap5
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off --weight-cap 3 --ckpt-dir ck_cap3

python scripts/test_all.py --ckpt-dir ck_cap5
```

Kỳ vọng: trần làm **precision lớp hiếm tăng, recall giảm nhẹ** — accuracy tăng,
macro-F1 có thể tăng hoặc giảm tuỳ mức đánh đổi. cap = 5 là điểm khởi đầu hợp lý;
nếu accuracy vẫn thấp thì thử cap = 3.

## Trạng thái framework

- Đặc trưng node: 29 · cạnh: 10
- skip_mode: `raw` (đã bỏ hybrid, 46.373 tham số)
- Loss: `cb_focal` + label smoothing 0,05 + **weight_cap 5,0**
- Resample: `off` (SMOTE đã loại) hoặc `weighted` với `--sampler-cap`
- Group DRO: **đã gỡ**
