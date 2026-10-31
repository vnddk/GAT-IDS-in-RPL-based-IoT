# Chiến lược cân bằng lớp — SMOTE bị loại, thay bằng lấy mẫu lại có trọng số

## Bằng chứng thực nghiệm: SMOTE GÂY HẠI

Chạy 120 epoch trên RADAR, cùng seed, chỉ khác `--resample`:

| Cấu hình | macro-F1 |
|---|---|
| `--resample smote` | 0,7829 |
| `--resample off`   | **0,8370** |

Chênh **+0,054**. Chi tiết theo lớp — 8 lớp tốt hơn, chỉ 3 lớp tệ hơn:

| Lớp | có SMOTE | bỏ SMOTE | Δ |
|---|---|---|---|
| delayed_reply | 0,165 | **0,559** | **+0,394** |
| hello_flood | 0,825 | 1,000 | +0,175 |
| worst_parent | 0,856 | 1,000 | +0,144 |
| selective_forward | 0,859 | 1,000 | +0,141 |
| sinkhole | 0,797 | 0,763 | −0,034 |
| continuous_sinkhole | 0,681 | 0,645 | −0,036 |

`delayed_reply` recall nhảy từ 0,101 lên 0,978. Node tổng hợp đã DẠY SAI mô hình —
đúng như phân tích trước đó: nội suy tuyến tính vi phạm quan hệ đại số giữa các
đặc trưng (x13 = |x0 − x8|), tạo ra mẫu không tương ứng trạng thái RPL nào có thật.

**SMOTE đã bị loại khỏi khuyến nghị.** Vẫn giữ trong code để tái lập thí nghiệm.

---

## Thay thế: lấy mẫu lại có trọng số ở mức đồ thị (`--resample weighted`)

### Vì sao vẫn cần, dù đã có CB-Focal?

CB-Focal chỉ tăng TRỌNG SỐ khi mẫu CÓ MẶT trong batch. Nó không tạo ra sự hiện
diện. Đo trên phân bố thật (18.266 đồ thị, batch 32):

| Lớp | Số đồ thị chứa | % batch KHÔNG có mẫu |
|---|---|---|
| replay | ~286 | **60,3%** |
| blackhole | ~302 | 58,7% |
| delayed_reply | ~619 | 33,2% |

Hơn một nửa số bước cập nhật hoàn toàn không thấy lớp hiếm nhất.

### Cách làm — KHÔNG sinh dữ liệu giả

Lấy mẫu ĐỒ THỊ THẬT có hoàn lại, xác suất tỉ lệ nghịch với độ hiếm:

    w(G) = ( N / n_{c*(G)} ) ^ alpha       c*(G) = lớp hiếm nhất trong G

Mọi mẫu đều là dữ liệu thật. Chỉ thay đổi TẦN SUẤT XUẤT HIỆN — không nội suy,
không bịa đặc trưng, không vi phạm quan hệ đại số. Khác hoàn toàn SMOTE.

### Hiệu quả đo được

| Lớp | shuffle thường | weighted α=0,5 | weighted α=1,0 |
|---|---|---|---|
| replay | 55,3% batch thiếu | **30,9%** | 11,7% |
| blackhole | 53,2% | **18,1%** | 10,6% |
| delayed_reply | 30,9% | **5,3%** | 11,7% |

Số node lớp hiếm thấy trong một epoch tăng ~6 lần (test riêng).

### Chọn alpha thế nào

- `alpha = 0,5` (mặc định) — ôn hoà. Tần suất kỳ vọng 0,13–2,81 lần/epoch.
- `alpha = 1,0` — cân bằng hẳn, nhưng tần suất min tụt xuống 0,01: một số đồ thị
  Normal gần như KHÔNG BAO GIỜ được thấy. Rủi ro làm hỏng lớp Normal.

**Cảnh báo cộng dồn:** CB-Focal đã bù một phần rồi. Dùng α = 1,0 cùng CB-Focal là
bù hai lần theo cấp số nhân — đúng lỗi mà SMOTE đã mắc. Nên bắt đầu từ 0,5.

---

## Chạy

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off      --ckpt-dir ck_off
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample weighted --sampler-alpha 0.5 --ckpt-dir ck_w05
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample weighted --sampler-alpha 1.0 --ckpt-dir ck_w10
```

Mốc so sánh: `--resample off` đã cho **0,8370**. Nếu `weighted` không vượt được
mốc này thì kết luận là CB-Focal đã đủ, và đó cũng là một kết quả đáng báo cáo.
