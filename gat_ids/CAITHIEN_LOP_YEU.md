# Cải thiện lớp yếu — chẩn đoán và giải pháp (v3.8)

## Ba phát hiện từ log 60 epoch

### 1. SMOTE ĐANG VÔ HIỆU HOÁ Class-Balanced Focal  ← nghiêm trọng nhất
Log ghi: `Loss long-tail: cb_focal | counts(min=7298,max=303012)`
Nhưng phân bố THẬT trước SMOTE có min = 573. SMOTE đã nâng lớp hiếm lên ~7000
TRƯỚC khi CB-Focal tính trọng số, nên nó tưởng dữ liệu đã cân bằng.

| | tỉ lệ w(hiếm)/w(Normal) |
|---|---|
| Đáng lẽ (phân bố gốc) | 18 lần |
| Thực tế (sau SMOTE)   | **2 lần** |

Hai cơ chế TRIỆT TIÊU NHAU. Đây là lỗi thiết kế, không phải lỗi tham số.
**ĐÃ SỬA**: `make_criterion(..., orig_counts=...)` dùng phân bố trước resample.

### 2. Mô hình DƯỚI-DỰ-ĐOÁN lớp tấn công
9/16 lớp có Precision > Recall, trung bình chênh +0,135:
    delayed_reply P=0,458 R=0,101 | sybil P=0,872 R=0,562 | hello P=1,000 R=0,701
Mô hình quá thận trọng. Khi P >> R, đánh đổi bớt precision lấy recall LÀM TĂNG F1.
**ĐÃ THÊM**: `scripts/calibrate_logits.py` — hiệu chỉnh logit sau huấn luyện,
dò τ trên VAL rồi áp cho TEST. Ước lượng bảo thủ +0,009 macro-F1.

### 3. Nối thêm skip đang LÀM GIẢM khả năng tách cụm
Silhouette: h1=0,0908 | h2=**0,2555** | final=0,2216
Tầng cuối (nối JK + raw-skip) tách KÉM HƠN h2 thuần 0,034.
=> Đề xuất "triple-skip" (nối thêm nữa) nhiều khả năng làm tệ hơn.

---

## 4 đặc trưng nhắm đích đã thêm (node 25 → 29)

Mỗi cái nhắm đúng chữ ký của một lớp yếu, kiểm chứng bằng Cohen's d:

| # | Đặc trưng | Nhắm lớp | Cơ chế | Cohen d |
|---|---|---|---|---|
| 21 | `rank_nunique` | clone_id | một ID công bố nhiều rank mâu thuẫn | 2,58 |
| 22 | `rank_per_hop` | rank | rank / (128 × độ sâu tô-pô) | **9,26** |
| 23 | `resp_delay_median` | delayed_reply | trung vị bền với ngoại lai hơn trung bình | 3,18 |
| 24 | `graph_n_nodes` | sybil | + #28 `n_nodes_ratio` so với lịch sử | 5,02 |

Ghi chú `resp_delay_median`: mean cho d=1,74, median cho d=3,18 — vì một gói trễ
bất thường kéo lệch trung bình. Trung vị là bản bền vững hơn của cùng tín hiệu.

Ghi chú `rank_per_hop`: cần ĐỘ SÂU tô-pô, tính bằng cách đi ngược chuỗi cha π(v)
tới gốc (có chặn chu trình, tối đa 32 hop).

---

## Chạy

```bash
# 1. Convert lại (log phải hiện d_n=29 d_e=10)
python scripts/convert_data.py --dataset radar --data-dir data/radar --window 5

# 2. PHÉP THỬ QUYẾT ĐỊNH: SMOTE có còn cần không sau khi sửa lỗi CB?
python scripts/train_all.py --models EdgeGAT --resample off   --ckpt-dir ck_nosmote
python scripts/train_all.py --models EdgeGAT --resample smote --ckpt-dir ck_smote

# 3. Hiệu chỉnh logit trên bản tốt hơn
python scripts/calibrate_logits.py --ckpt-dir ck_nosmote
```

Bước 2 quan trọng: trước đây SMOTE che mất tác dụng của CB-Focal nên không thể
biết cái nào có ích. Giờ sửa rồi, hai cấu hình mới so sánh được công bằng.
