# Bảng lệnh chạy — FedEdge-GAT-IDS v3.8

## Bước 1 — Convert (chạy MỘT lần, dùng chung cho mọi model)
```bash
python scripts/convert_data.py --dataset radar --data-dir data/radar --window 5
```
Kiểm 3 thứ trong log: `d_n=29 d_e=10` · kiểm toán báo "Không phát hiện rò rỉ" ·
bảng phân bố node đủ 16 lớp ở cột test.

---

## Bước 2 — Huấn luyện

### Cách A — chạy CẢ 6 model một lệnh (khuyến nghị cho bảng so sánh cuối)
```bash
python scripts/train_all.py --epochs 120
```

### Cách B — chạy RIÊNG từng model
Tên hợp lệ: `RF`, `XGBoost`, `MLP`, `GCN`, `E-GraphSAGE`, `EdgeGAT`

```bash
# Mô hình đề xuất
python scripts/train_all.py --epochs 120 --models EdgeGAT

# Baseline cây (rất nhanh, ~vài phút; --epochs không ảnh hưởng)
python scripts/train_all.py --models RF
python scripts/train_all.py --models XGBoost

# Baseline mạng nơ-ron
python scripts/train_all.py --epochs 120 --models MLP
python scripts/train_all.py --epochs 120 --models GCN
python scripts/train_all.py --epochs 120 --models E-GraphSAGE

# Nhiều model một lệnh
python scripts/train_all.py --epochs 120 --models RF,XGBoost
python scripts/train_all.py --epochs 120 --models GCN,E-GraphSAGE,EdgeGAT
```

QUAN TRỌNG: các lệnh riêng lẻ dùng CÙNG `--ckpt-dir` (mặc định `checkpoints/`)
sẽ TÍCH LUỸ vào cùng thư mục, mỗi model một thư mục con. Sau đó `test_all.py`
gộp tất cả thành một bảng. Không cần chạy lại model đã xong.

---

## Bước 3 — Kiểm thử (gộp bảng so sánh)
```bash
python scripts/test_all.py
```
Bỏ qua model chưa có checkpoint, chỉ báo cáo model đã huấn luyện.

---

## Các cờ hữu ích

| Cờ | Ý nghĩa |
|---|---|
| `--models` | Chọn model: `EdgeGAT`, `RF,XGBoost`, `all` (mặc định) |
| `--ckpt-dir` | Thư mục lưu (tách riêng để không đè lần chạy khác) |
| `--epochs` | Số epoch (chỉ ảnh hưởng model nơ-ron) |
| `--resample` | `off` (KHUYẾN NGHỊ) \| `weighted` \| `smote` (bỏ) \| `balance` \| `ratio` |
| `--sampler-alpha` | Độ mạnh của `weighted`: 0,5 ôn hoà (mặc định), 1,0 cân bằng hẳn |
| `--skip-mode` | `none` \| `raw` \| `hybrid` — CHỈ tác động EdgeGAT |

---

## Các thí nghiệm cho luận văn

### TN-1 — Bảng so sánh 6 model (kết quả chính, Chương 4)
```bash
python scripts/train_all.py --epochs 120 --ckpt-dir ck_main
python scripts/test_all.py --ckpt-dir ck_main
```

### TN-2 — Chiến lược cân bằng lớp  (SMOTE ĐÃ BỊ LOẠI, xem kết quả bên dưới)
Bằng chứng thực nghiệm 120 epoch: SMOTE 0,7829  vs  không SMOTE **0,8370** (+0,054).
SMOTE đã bị loại khỏi khuyến nghị. So sánh còn lại:
```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off      --ckpt-dir ck_off
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample weighted --sampler-alpha 0.5 --ckpt-dir ck_w05
python scripts/train_all.py --epochs 120 --models EdgeGAT --resample weighted --sampler-alpha 1.0 --ckpt-dir ck_w10
python scripts/test_all.py --ckpt-dir ck_off
python scripts/test_all.py --ckpt-dir ck_w05
python scripts/test_all.py --ckpt-dir ck_w10
```

### TN-3 — Ablation skip connection (chứng minh đóng góp riêng)
```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --skip-mode none   --ckpt-dir ck_skip_none
python scripts/train_all.py --epochs 120 --models EdgeGAT --skip-mode raw    --ckpt-dir ck_skip_raw
python scripts/train_all.py --epochs 120 --models EdgeGAT --skip-mode hybrid --ckpt-dir ck_skip_hybrid
```
Lưu ý cho hội đồng: baseline KHÔNG có raw-skip, nên nếu chỉ so EdgeGAT(raw/hybrid)
với GCN thì không tách được "công của attention" khỏi "công của skip". Cấu hình
`--skip-mode none` chính là câu trả lời: nếu EdgeGAT không skip vẫn hơn GCN thì
attention có giá trị riêng.

### TN-4 — Ablation đặc trưng cạnh (trả lời KC-2)
```bash
python scripts/train.py --no-edge --epochs 120 --ckpt-dir ck_noedge
```

### TN-5 — Hiệu chỉnh logit (nâng recall lớp hiếm, không train lại)
```bash
python scripts/calibrate_logits.py --ckpt-dir ck_main
```

### TN-6 — Nhiều seed (BẮT BUỘC cho luận văn)
```bash
for s in 42 43 44; do
  python scripts/train_all.py --epochs 120 --models EdgeGAT --ckpt-dir ck_seed$s
  python scripts/test_all.py --ckpt-dir ck_seed$s
done
```
Windows CMD:
```cmd
for %s in (42 43 44) do python scripts/train_all.py --epochs 120 --models EdgeGAT --ckpt-dir ck_seed%s
```
Báo cáo trung bình ± độ lệch chuẩn, không phải một con số đơn lẻ.

### TN-7 — Trực quan hoá t-SNE
```bash
python scripts/tsne_layers.py --ckpt-dir ck_main --n-samples 4000
python scripts/tsne_layers.py --ckpt-dir ck_main --exclude-normal
```
