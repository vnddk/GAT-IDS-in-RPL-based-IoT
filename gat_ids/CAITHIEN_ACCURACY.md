# Cải thiện Accuracy — phân tích ngân sách lỗi và các núm xoay

## Hiện trạng: mô hình đã LẬT từ dưới-dự-đoán sang BÙ QUÁ ĐÀ

| | có SMOTE | bỏ SMOTE |
|---|---|---|
| Accuracy | 0,9345 | 0,9301 |
| Macro-F1 | 0,7829 | **0,8370** |

Macro-F1 tăng nhưng accuracy GIẢM nhẹ. Nhìn per-class thấy rõ nguyên nhân:

| Lớp | P | R | Chẩn đoán |
|---|---|---|---|
| delayed_reply | 0,391 | 0,978 | HÚT nhầm 408 node |
| version | 0,680 | 0,992 | hút nhầm 62 |
| sinkhole | 0,676 | 0,875 | hút nhầm 433 |
| clone_id | 0,568 | 0,735 | hút nhầm 881 |

Trước khi bỏ SMOTE thì recall thấp (bỏ sót). Giờ ngược lại: recall cao, precision
thấp — các lớp hiếm đang hút node của lớp khác vào. **CB-Focal với counts đúng
đang bù quá đà.**

## Ngân sách lỗi (5.194 lỗi / 74.259 node)

| Nguồn | Số node bị hút nhầm vào |
|---|---|
| Normal | 2.333 |
| clone_id | 881 |
| sinkhole | 433 |
| rank | 428 |
| delayed_reply | 408 |
| continuous_sinkhole | 337 |

Normal chiếm 87,6% tập test, nên **mỗi 1% recall Normal = +0,0088 accuracy**:

| Normal recall | Accuracy |
|---|---|
| 0,960 (hiện tại) | 0,9301 |
| 0,975 | 0,9432 |
| 0,985 | 0,9520 |
| 0,990 | 0,9563 |

---

## NÚM XOAY 1 — `--cb-beta` (mạnh nhất, phải train lại)

beta điều khiển độ mạnh hiệu chỉnh mất cân bằng:

| beta | tỉ lệ w(hiếm)/w(Normal) | tác dụng |
|---|---|---|
| 0,99 | 1,4× | gần như tắt hiệu chỉnh — accuracy cao nhất |
| 0,999 | 8,2× | trung dung |
| 0,9999 (đang dùng) | 77,4× | ưu tiên macro-F1 |

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --cb-beta 0.999 --ckpt-dir ck_b999
python scripts/train_all.py --epochs 120 --models EdgeGAT --cb-beta 0.99  --ckpt-dir ck_b99
```

## NÚM XOAY 2 — hiệu chỉnh logit τ ÂM (rẻ nhất, KHÔNG train lại)

Vì mô hình đang bù quá đà, τ < 0 sẽ kéo về, tăng accuracy:

```bash
python scripts/calibrate_logits.py --ckpt-dir ck_off --metric accuracy
python scripts/calibrate_logits.py --ckpt-dir ck_off --metric balanced   # cân bằng cả hai
```

τ dò trên VAL, áp cho TEST. Chạy trong vài phút, không cần huấn luyện lại.

---

## NÊN LƯỢC BỎ GÌ

### 1. Ba đặc trưng version — CHẮC CHẮN bỏ
Kiểm toán cho nMI ≈ 0,001 và λ chạm biên −2,0 (hai phép chẩn đoán độc lập cùng
kết luận). Nguyên nhân: cột `RPL_VERSION` trong RADAR là HẰNG SỐ (đã kiểm tra
trên file thật: chỉ có giá trị 0, std = 0).

    version_mean (#2), nbr_version_avg (#9), diff_version (#16)

Bỏ 3 cột này: d_n 29 → 26. Ít nhiễu hơn, ít tham số hơn, không mất thông tin.
Chạy `python KIEMTRA_RPL_VERSION.py --raw-dir data/radar` để xác nhận lần cuối.

### 2. SMOTE — ĐÃ bỏ (bằng chứng +0,054 macro-F1)

### 3. Nhánh hybrid skip — CÂN NHẮC bỏ
Silhouette: h2 = 0,2555 nhưng final = 0,2216. Nối JK + raw-skip làm GIẢM khả
năng tách cụm 0,034. Chạy ablation `--skip-mode raw` và `--skip-mode none` để
quyết định bằng số, đừng giữ chỉ vì đã cài.

### 4. KHÔNG bỏ: CB-Focal, EMA, Yeo-Johnson, 4 đặc trưng thời gian
Đều có bằng chứng đóng góp rõ ràng.

---

## Thứ tự khuyến nghị

1. `--metric balanced` trên checkpoint hiện có — vài phút, xem ngay được đánh đổi
2. Bỏ 3 đặc trưng version, convert lại
3. `--cb-beta 0.999` — điểm trung dung, nhiều khả năng cho cả accuracy lẫn
   macro-F1 tốt hơn hiện tại
4. Ablation `--skip-mode` để quyết định giữ hay bỏ hybrid

## Lưu ý cho luận văn

Accuracy và macro-F1 ĐÁNH ĐỔI trực tiếp trên dữ liệu mất cân bằng 87,6%. Không
có cấu hình nào tối ưu cả hai. Với IDS, **macro-F1 là chỉ số đúng** — một hệ
thống bỏ sót tấn công hiếm nhưng đạt accuracy 0,99 là vô dụng. Nên báo cáo cả
hai, nêu rõ vì sao ưu tiên macro-F1, và trình bày đường cong đánh đổi qua các
giá trị beta như một đóng góp phân tích.
