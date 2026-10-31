# Lệnh chạy — GAT-Transformer-IDS trên UOS, và các thí nghiệm giải đáp thắc mắc

Phiên bản framework: **v3.40** (sửa sụp biểu diễn SSL + GAT-Transformer vào
`test_all.py` + cờ ablation `--no-shuffle`).

---

## 0. Ba thay đổi trong bản này

| # | Tệp | Thay đổi | Vì sao |
|---|---|---|---|
| 1 | `scripts/train_ssl.py` | Thêm **stop-gradient** ở nhánh đích: `mse_loss(hr, h.detach())` | Mã cũ để `mse_loss(hr, h)` với `h` còn gradient → bộ mã hoá tự co nhỏ `h` về 0, MSE → 0 mà không học được gì (**sụp biểu diễn**). Đây là lý do nhật ký RADAR có `MSE 0.00000` suốt 50 epoch nhưng ROC-AUC chỉ 0,6731 |
| 2 | `scripts/train_ssl.py` | Ghi thêm `var(h)` và **MSE tương đối** vào nhật ký; cảnh báo khi `var(h) < 1e-6`; lưu **đầy đủ** siêu tham số vào `model_config.json` | MSE tuyệt đối không phát hiện được sụp; MSE/var(h) thì có. Bản ghi cấu hình cũ thiếu 12 khoá nên `test_all.py` không dựng lại được mô hình |
| 3 | `scripts/test_all.py` | Nhận diện `GAT-Transformer`; thêm `--gt-dir`, `--skip`; **bọc `mode="clf"`** khi suy luận | `forward` mặc định `mode="ssl"` trả về cặp `(h, h_rec)` chứ không phải logits — nếu không bọc thì `argmax` chạy trên biểu diễn tái tạo và **sai âm thầm, không báo lỗi** |
| 4 | `scripts/train_ata.py` | Thêm cờ ablation `--no-shuffle` | Để đo trực tiếp xem thứ tự lô có ảnh hưởng gì không (xem mục 3) |

---

## 1. Chạy GAT-Transformer-IDS trên bộ UOS

### 1.1 Chuẩn bị dữ liệu (nếu chưa có)

```bash
python scripts/convert_data.py --dataset uos --split chrono --split-by-file \
       --out-dir data/uos_chrono
```

### 1.2 Giai đoạn 1 — tiền huấn luyện tự giám sát

```bash
python scripts/train_ssl.py --data-dir data/uos_chrono --stage ssl \
       --backbone negat --epochs 60 --ckpt-dir ck_ssl_uos \
       --d-model 32 --dim-ff 64 --tr-layers 1
```

> **Đọc nhật ký thế nào.** Bản mới in thêm hai cột:
> ```
> Epoch  12 | MSE train 0.00031 | MSE val 0.00028 | var(h) 0.04120 | MSE tương đối 0.0068
> ```
> - `MSE tương đối` = `MSE / var(h)`, **bất biến với phép co giãn `h`**. Đây mới
>   là chỉ số đáng tin. MSE tuyệt đối nhỏ mà MSE tương đối không nhỏ nghĩa là
>   mô hình đang co nhỏ tín hiệu chứ không tái tạo tốt hơn.
> - `var(h)` phải **ổn định hoặc tăng**. Nếu nó giảm dần về 0 thì biểu diễn
>   đang sụp và script sẽ tự cảnh báo.

**Nếu UOS báo lỗi "Không còn đồ thị nào sau khi lọc":** bộ UOS mất cân bằng theo
chiều ngược (gần như mọi cửa sổ đều chứa nút tấn công), nên không đủ đồ thị
thuần bình thường để tiền huấn luyện. Bản mới in ra hai lối đi; đơn giản nhất là
bỏ giai đoạn SSL và chạy thẳng:

```bash
python scripts/train_ssl.py --data-dir data/uos_chrono --stage scratch \
       --backbone negat --epochs 120 --ckpt-dir ck_gtids_uos \
       --d-model 32 --dim-ff 64 --tr-layers 1
```

### 1.3 Giai đoạn 2 — tinh chỉnh có giám sát

```bash
python scripts/train_ssl.py --data-dir data/uos_chrono --stage finetune \
       --backbone negat --init-from ck_ssl_uos --epochs 120 \
       --ckpt-dir ck_gtids_uos --d-model 32 --dim-ff 64 --tr-layers 1
```

### 1.4 Đưa vào bảng so sánh chung

```bash
python scripts/test_all.py --data-dir data/uos_chrono \
       --gt-dir ck_gtids_uos --binary
```

`--gt-dir` trỏ tới thư mục checkpoint **sau khi tinh chỉnh**. Bỏ trống thì
script tìm ở `checkpoints/GAT-Transformer` (là thư mục mặc định của
`train_ssl.py`).

Làm y hệt cho RADAR:

```bash
python scripts/test_all.py --data-dir data/radar_chrono \
       --gt-dir ck_gtids_radar --binary
```

> **Lưu ý về lỗi cũ trong nhật ký anh gửi.** Lần chạy thứ hai báo
> `Không nhận diện model type: GAT-Transformer` rồi `Bỏ qua NE-GAT`. Nguyên
> nhân: thư mục `checkpoints/NE-GAT/` đã bị ghi đè bởi một checkpoint
> GAT-Transformer. Bản mới in kèm **đường dẫn thư mục** trong thông báo lỗi để
> lần sau thấy ngay nhầm ở đâu. Nên giữ mỗi mô hình một thư mục riêng.

---

## 2. Thí nghiệm QUYẾT ĐỊNH cho GAT-Transformer

Đây là thí nghiệm nên chạy trước mọi thứ khác, vì nó trả lời câu hỏi
*"tiền huấn luyện tự giám sát có đóng góp gì không?"*

```bash
# (A) đối chứng — huấn luyện có giám sát từ đầu, KHÔNG tiền huấn luyện
python scripts/train_ssl.py --data-dir data/radar_chrono --stage scratch \
       --backbone negat --epochs 120 --ckpt-dir ck_gt_scratch_radar \
       --d-model 32 --dim-ff 64 --tr-layers 1 --seed 42

# (B) tiền huấn luyện ĐÃ SỬA rồi tinh chỉnh
python scripts/train_ssl.py --data-dir data/radar_chrono --stage ssl \
       --backbone negat --epochs 60 --ckpt-dir ck_ssl_radar_v2 \
       --d-model 32 --dim-ff 64 --tr-layers 1 --seed 42

python scripts/train_ssl.py --data-dir data/radar_chrono --stage finetune \
       --backbone negat --init-from ck_ssl_radar_v2 --epochs 120 \
       --ckpt-dir ck_gt_ft_radar_v2 --d-model 32 --dim-ff 64 --tr-layers 1 --seed 42

# (C) tái hiện lỗi cũ để đối chứng — CHỈ để chứng minh, không dùng làm kết quả
python scripts/train_ssl.py --data-dir data/radar_chrono --stage ssl \
       --backbone negat --epochs 60 --ckpt-dir ck_ssl_radar_collapse \
       --d-model 32 --dim-ff 64 --tr-layers 1 --no-stopgrad --seed 42
```

**Cách đọc kết quả:**

| So sánh | Kết luận |
|---|---|
| (B) > (A) rõ rệt | Tiền huấn luyện có giá trị → giữ kiến trúc hai giai đoạn, viết vào luận văn |
| (B) ≈ (A) | Tiền huấn luyện **không đóng góp gì**; GAT-Transformer chỉ là NE-GAT cộng thêm 44% tham số. Phải nói thẳng điều này |
| ROC-AUC của (B) > ROC-AUC của (C) = 0,6731 | Xác nhận sụp biểu diễn là có thật và đã sửa được |

---

## 3. Trả lời thắc mắc về `shuffle=True`

### 3.1 Đo trực tiếp thay vì tranh luận

```bash
# lấy ngẫu nhiên trong mỗi miền — đúng Thuật toán 1
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 3 --lambda-d 0.1 --epochs 120 --ckpt-dir ck_ata_shuf --seed 42

# duyệt đúng thứ tự thời gian trong mỗi miền — ABLATION
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 3 --lambda-d 0.1 --epochs 120 --ckpt-dir ck_ata_noshuf \
       --no-shuffle --seed 42
```

### 3.2 Kiểm tra trường hợp suy biến K=1

```bash
# ATA tắt hoàn toàn — phải cho kết quả TRÙNG với train_all.py
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 1 --lambda-d 0 --epochs 120 --ckpt-dir ck_ata_k1 --seed 42
```

Nếu `ck_ata_k1` khác đáng kể kết quả `train_all.py` với cùng hạt giống, nghĩa là
còn một yếu tố nhiễu nào đó ngoài ATA giữa hai đường huấn luyện — lúc đó mọi
phép so sánh ATA / không-ATA đều không dùng được. Đây là **phép kiểm tra sức
khoẻ** nên chạy một lần.

---

## 4. Việc còn thiếu, theo thứ tự ưu tiên

1. **Chạy ≥5 hạt giống** cho mọi cấu hình chính. Chênh lệch giữa NE-GAT
   (0,9440) và GAT-Transformer (0,9430) là **0,0010** — hiện **không phân biệt
   được với nhiễu**. Không có thanh sai số thì không kết luận được mô hình nào
   tốt hơn.

   ```bash
   for s in 1 2 3 4 5; do
     python scripts/train_all.py --data-dir data/radar_chrono \
            --ckpt-dir ck_seed$s --seed $s
   done
   python scripts/aggregate_seeds.py --dirs ck_seed1 ck_seed2 ck_seed3 ck_seed4 ck_seed5
   ```

2. **Chạy lại ablation `--no-edge`** với mã đã vá `allow_abbrev=False`.
3. **Thí nghiệm hiệu quả nhãn** ở 10/25/50/100% nhãn.
