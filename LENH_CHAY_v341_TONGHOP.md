# Bảng lệnh tổng hợp — framework v3.41 (đã rà soát 2026-10-04)

Tệp này thay thế việc phải mở sáu tệp `LENH_*.md` rời rạc. Mọi lệnh dưới đây
**đã được chạy thật** trên bộ RPLGuard và RPL-Beh trong lần rà soát này, trừ
những chỗ ghi rõ là *chưa kiểm chứng được* vì thiếu dữ liệu thô.

---

## 0. Chuẩn bị môi trường

```bash
pip install --break-system-packages xgboost lightgbm
```

**Bắt buộc.** `src/models/tabular_baselines.py` import `xgboost` ở mức
module, nên thiếu gói này thì `train_all.py` và `test_all.py` chết ngay ở
bước import — không phải chỉ bỏ qua ba mô hình cây. Phiên bản đã kiểm chứng:
xgboost 3.2.0, lightgbm 4.7.0, torch 2.14.0, PyG 2.8.0.

---

## 1. Ba bước xương sống — áp dụng cho MỌI bộ dữ liệu

| Bước | Lệnh | Sinh ra |
|---|---|---|
| 1 | `scripts/convert_data.py` | `train.pt`, `val.pt`, `test.pt`, `scaler.pt`, `meta.json` |
| 2 | `scripts/train_all.py` | `<ckpt>/<Mô hình>/best_model.pt` + `model_config.json` (hoặc `model.pkl` cho cây) |
| 3 | `scripts/test_all.py` | `test_comparison.json` + bảng xếp hạng |
| 4 | `scripts/run_all_viz.py` | `cm_*.png/.csv` + `tsne_*.png` + `viz_summary_*.json` |

---

## 2. RPLGuard — bộ mới nhất

### 2.1 Chuyển đổi

```bash
python scripts/convert_data.py \
    --dataset rplguard \
    --data-dir /duong/dan/RPLGuard_Dataset-v1-2025 \
    --window 5 \
    --split chrono \
    --out-dir data/rplguard_chrono
```

Kết quả đã đo: **6 tệp → 76.468 dòng → 433 đồ thị**, `d_n=13`, `d_e=9`,
3 lớp, chia `train=301 / val=42 / test=90`, δ̄ = 15,5 cửa sổ.

**Dùng `--split chrono`, KHÔNG dùng `--split-by-file`.** Chia theo tệp làm
`val` mất sạch lớp *greyhole* và `test` mất sạch lớp *blackhole*, nên các chỉ
số macro trở nên vô nghĩa. `chrono` giữ đủ cả ba lớp ở cả ba tập.

Hai biến thể:

```bash
# nhãn nhị phân (Normal / attack)
python scripts/convert_data.py --dataset rplguard_binary ... --out-dir data/rplguard_bin

# tái hiện rò rỉ rank để đối chứng — KHÔNG dùng cho kết quả chính thức
RPLGUARD_RANK_MODE=raw python scripts/convert_data.py --dataset rplguard ...
```

`rank_mode` mặc định là `drop`. Giá trị hợp lệ: `drop` | `relative` | `raw`,
truyền qua **biến môi trường** `RPLGUARD_RANK_MODE`, không phải cờ dòng lệnh.

### 2.2 Huấn luyện và kiểm thử

```bash
python scripts/train_all.py \
    --data-dir data/rplguard_chrono \
    --ckpt-dir checkpoints_rplguard \
    --epochs 80

python scripts/test_all.py \
    --data-dir data/rplguard_chrono \
    --ckpt-dir checkpoints_rplguard
```

> **`--epochs 80`, đừng để 12.** Đây là phát hiện định lượng của lần rà soát
> này. Cùng một dữ liệu, cùng một hạt giống, chỉ khác số vòng:
>
> | Mô hình | 12 vòng | 80 vòng |
> |---|---|---|
> | EdgeGAT-IDS | 0,6399 | **0,9950** |
> | NE-GAT-IDS | 0,8528 | **0,9863** |
>
> Ở 12 vòng, EdgeGAT xếp **thứ 7/8** và trông như một mô hình hỏng; ở 80 vòng
> nó xếp **thứ 1**, vượt cả LightGBM (0,9915). Nếu báo cáo con số 12 vòng thì
> kết luận "mô hình đề xuất kém hơn cây quyết định" là một kết luận SAI do
> ngân sách huấn luyện, không phải do kiến trúc.

Bảng đã đo ở 12 vòng (để đối chiếu, **không** phải kết quả báo cáo):

```
  RF          0.9451      MLP          0.7563
  XGBoost     0.9949      GCN          0.5724
  LightGBM    0.9963      E-GraphSAGE  0.8589
  EdgeGAT     0.6399      NE-GAT       0.8528
```

---

## 3. t-SNE và ma trận nhầm lẫn — mọi mô hình, mọi bộ

### 3.1 Một bộ dữ liệu

```bash
python scripts/run_all_viz.py \
    --data-dir data/rplguard_chrono \
    --ckpt-dir checkpoints_rplguard \
    --out-dir viz/rplguard \
    --tag rplguard \
    --n-samples 3000
```

Hành vi (đã kiểm chứng trên RPLGuard, 8/8 mô hình đã huấn luyện đều ra hình):

| Mô hình | t-SNE | Ghi chú |
|---|---|---|
| EdgeGAT-IDS, NE-GAT-IDS | **từng tầng** | `tầng 1`, `tầng 2`, `trước bộ phân loại` — số bảng = `num_layers` + 1 |
| RF, XGBoost, LightGBM | tầng cuối | véc-tơ xác suất lớp đóng vai biểu diễn cuối |
| MLP, GCN, E-GraphSAGE, GAT-Transformer | tầng cuối | |
| Tất cả | — | ma trận nhầm lẫn, không trừ mô hình nào |

Giới hạn danh sách mô hình — **nhận cả hai cú pháp** (đã sửa trong lần rà soát):

```bash
python scripts/run_all_viz.py ... --models EdgeGAT NE-GAT        # dấu cách
python scripts/run_all_viz.py ... --models EdgeGAT,NE-GAT        # dấu phẩy
```

Cờ phụ: `--skip-tsne`, `--skip-cm`, `--split train|val|test`,
`--perplexity`, `--gt-dir <thư mục GAT-Transformer sau tinh chỉnh>`.

### 3.2 Cả bốn bộ trong một lần

```bash
DATA_RADAR=data/radar_chrono   CKPT_RADAR=checkpoints_radar \
DATA_UOS=data/uos_chrono       CKPT_UOS=checkpoints_uos \
DATA_BEH=data/rplbeh_chrono    CKPT_BEH=checkpoints_rplbeh \
DATA_GUARD=data/rplguard_chrono CKPT_GUARD=checkpoints_rplguard \
GT_RADAR=ck_gtids_radar        GT_UOS=ck_gtids_uos \
OUT=viz NS=3000 \
bash scripts/run_all_viz_4ds.sh
```

Script tự bỏ qua bộ nào chưa có `meta.json` và in rõ lý do.

### 3.3 Đọc dòng đối chiếu ở cuối

```
── đối chiếu: yêu cầu 9 | thành công 8 | chưa huấn luyện 1 | LỖI 0
chưa huấn luyện: GAT-Transformer
```

Dòng này là **phần thêm của lần rà soát**. `LỖI` khác hẳn `chưa huấn luyện`:
`chưa huấn luyện` là thiếu checkpoint (bình thường), còn `LỖI` là lỗi mã phải
sửa, và nay in kèm cả vết gọi. Script trả mã thoát 1 nếu có `LỖI`, nên dùng
được trong CI.

---

## 4. Ba bộ còn lại

```bash
# RADAR
python scripts/convert_data.py --dataset radar --data-dir data/radar \
       --window 5 --split chrono --out-dir data/radar_chrono
python scripts/train_all.py --data-dir data/radar_chrono --ckpt-dir checkpoints_radar --epochs 80
python scripts/test_all.py  --data-dir data/radar_chrono --ckpt-dir checkpoints_radar

# UOS — chia theo kịch bản hoặc theo tệp
python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
       --split-by-file "0,1,2,3,4:5,6:7,8,9" --split-key src_file \
       --out-dir data/uos_byfile
python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
       --split chrono --out-dir data/uos_chrono

# RPL-Beh — cửa sổ do bộ dữ liệu ấn định (480 s), --window bị bỏ qua
python scripts/convert_data.py --dataset rplbeh --data-dir data/rplbeh \
       --split chrono --out-dir data/rplbeh_chrono
```

Đã đo cho RPL-Beh: **158.254 dòng → 7.030 đồ thị**, `d_n=24`, `d_e=10`,
5 lớp, `train=4845 / val=665 / test=1520` trên 95 tệp, δ̄ = 124,0 cửa sổ.

---

## 5. GAT-Transformer-IDS — hai giai đoạn

```bash
# Giai đoạn 1 — tiền huấn luyện tự giám sát (chỉ nút bình thường)
python scripts/train_ssl.py --stage ssl \
    --data-dir data/radar_chrono --ckpt-dir ck_gtids_ssl \
    --backbone negat --epochs 60 --d-model 32 --mask-ratio 0.3

# Giai đoạn 2 — tinh chỉnh có giám sát
python scripts/train_ssl.py --stage finetune \
    --data-dir data/radar_chrono --ckpt-dir ck_gtids_radar \
    --init-from ck_gtids_ssl --epochs 60

# ĐỐI CHỨNG BẮT BUỘC — cùng kiến trúc, KHÔNG tiền huấn luyện
python scripts/train_ssl.py --stage scratch \
    --data-dir data/radar_chrono --ckpt-dir ck_gtids_scratch --epochs 60

# Đưa vào bảng so sánh và vào hình
python scripts/test_all.py    --data-dir data/radar_chrono --ckpt-dir checkpoints_radar \
       --gt-dir ck_gtids_radar
python scripts/run_all_viz.py --data-dir data/radar_chrono --ckpt-dir checkpoints_radar \
       --out-dir viz/radar --tag radar --gt-dir ck_gtids_radar
```

> **`--d-model 32` không phải tuỳ ý.** Phải thoả `d_model ≤ d_gnn`. Vi phạm
> điều kiện này biến bộ tự mã hoá thành hàm đồng nhất: tái tạo hoàn hảo mà
> không học được gì, đo được ROC-AUC = 0,3813 — **thấp hơn ngẫu nhiên**.
>
> **Không có `--stage scratch` thì không kết luận được gì về ích lợi của tiền
> huấn luyện.** Đây vẫn là thí nghiệm còn thiếu.

---

## 6. ATA — căn chỉnh thời gian thích nghi

```bash
python scripts/train_ata.py --arch negat \
    --data-dir data/radar_chrono --ckpt-dir ck_ata_negat \
    --disc coral --domain-strategy adjacent --epochs 80

# ablation: tắt trộn ngẫu nhiên để kiểm chứng lập luận trình tự thời gian
python scripts/train_ata.py --arch negat ... --no-shuffle
```

`train_ata.py` và `train_ssl.py` là **hai script duy nhất** đặt
`allow_abbrev=False`. Mọi script khác vẫn cho phép viết tắt cờ, nên
`--edge-l`, `--no-edge-u`, `--resam` sẽ được **âm thầm** chấp nhận ở
`train_all.py` và có thể khớp sai cờ. Luôn viết đủ tên cờ.

---

## 7. Nhiều hạt giống

```bash
for s in 1 2 3 4 5; do
  python scripts/train_all.py --data-dir data/rplguard_chrono \
      --ckpt-dir ck_seed$s --models EdgeGAT,NE-GAT --epochs 80 --seed $s
done

python scripts/aggregate_seeds.py --dirs ck_seed1,ck_seed2,ck_seed3,ck_seed4,ck_seed5 \
       --model EdgeGAT
```

`--dirs` nhận **một chuỗi phân tách bằng dấu phẩy**, không nhận danh sách cách
nhau bởi dấu cách.

---

## 8. Lỗi trong các tệp lệnh cũ — đừng sao chép

Rà soát đối chiếu từng dòng lệnh trong các tệp `.md` với `argparse` thật:

| Tệp | Lệnh sai | Vì sao |
|---|---|---|
| `LENH_CHAY_ATA.md` dòng 243 | `aggregate_seeds.py --pattern "kq_*.json"` | không có cờ `--pattern`; cờ bắt buộc là `--dirs`, và script đọc **thư mục checkpoint**, không gom tệp json |
| `LENH_CHAY_ATA.md` dòng 153, 155 | `train_ata.py --arch negat --edge-agg mean ... --ckpt-dir ...` | dấu `...` là tham số vị trí lạ → `unrecognized arguments: ...` |
| `LENH_CHAY_GATTRANSFORMER.md` dòng 24 | `convert_data.py --split-by-file --out-dir ...` | `--split-by-file` **nhận giá trị**, không phải cờ bật/tắt → `expected one argument` |
| `LENH_CHAY_GATTRANSFORMER.md` dòng 169 | `aggregate_seeds.py --dirs ck_seed1 ck_seed2 ...` | `--dirs` không có `nargs`; phải dùng dấu phẩy |

Tên bộ dữ liệu, tên mô hình, đường dẫn script và mọi giá trị `choices=` khác
trong các tệp đó đều đã kiểm tra và **đúng**.

---

## 9. Những gì KHÔNG chạy được trong v3.41

### 9.1 T-GAT-IDS — không có script huấn luyện

`src/models/temporal_gat.py` tồn tại và khởi tạo được (98.827 tham số ở chiều
RPLGuard, forward một ảnh chụp cho logits đúng dạng), nhưng:

- không có `scripts/train_temporal.py`;
- không nằm trong `_ALL` của `train_all.py`, trong danh sách của `test_all.py`,
  hay trong `ALL_MODELS` của `run_all_viz.py`.

Nên **không có lệnh nào chạy được T-GAT-IDS**. Nó phải được nêu là hướng phát
triển, không phải là kết quả.

### 9.2 Ba bộ thiếu `node_ids` — đã sửa trong lần rà soát

`build_node_sequences` cần `node_ids` để nối cùng một thiết bị qua các cửa sổ.
Trước khi sửa, chỉ RADAR có trường này:

| Bộ | Trước | Sau |
|---|---|---|
| RADAR | có | có |
| RPLGuard | **0/90 đồ thị** | 90/90 |
| RPL-Beh | **0/1520 đồ thị** | 1520/1520 |
| UOS | không có | có (CRC-32 của địa chỉ IPv6) |

Thiếu trường này, hàm rơi về `torch.arange(num_nodes)` — ghép theo **vị trí**
trong danh sách nút đã sắp xếp cục bộ. Số nút mỗi cửa sổ của RPLGuard dao động
mạnh (đo được: 9, 8, 4, 9, 4, 4, 9, 3, 6, 8, 4, 7), nên vị trí *k* ở cửa sổ *t*
là một thiết bị **khác** với vị trí *k* ở cửa sổ *t+1*.

Lỗi này **không gây ngoại lệ**: nó sinh ra tensor hợp lệ với tỉ lệ che 0,27 và
trông hoàn toàn bình thường. Sau khi sửa, tỉ lệ che là 0,34 — con số **cao hơn
mới là con số đúng**, vì việc đối chiếu thiết bị thật phát hiện ra những lần
nút thực sự vắng mặt mà phép ghép theo vị trí đã lấp bừa.

---

## 10. Hai lỗi đã sửa trong `run_all_viz.py`

Cả ba mô hình cây trước đây **không sinh ra ma trận nhầm lẫn lẫn t-SNE** trên
mọi bộ dữ liệu, và chỉ in `bỏ qua (chưa huấn luyện hoặc thiếu tệp)` dù
checkpoint tồn tại.

**Lỗi 1 — lệch không gian đặc trưng.** `train_all.py` huấn luyện cây trên
`_flatten_graphs`, nối `[x ‖ trung bình láng giềng ‖ cực đại láng giềng]` nên
số cột là `3·d_n`. `_tree_forward` lại dùng trực tiếp `g.x`:

```
ValueError: X has 13 features, but RandomForestClassifier is expecting 39 features
```

**Lỗi 2 — checkpoint chưa bóc vỏ.** XGBoost và LightGBM được lưu dưới dạng
`{"model": clf, "remap": remap}` (hai thư viện đòi nhãn liên tục 0..K-1), còn
RF lưu trực tiếp bộ phân loại. `_load_tree` trả nguyên cái dict:

```
AttributeError: 'dict' object has no attribute 'predict'
```

Cả hai lỗi bị `try/except` ở vòng ngoài nuốt mất và chỉ in một dòng cảnh báo,
nên suốt nhiều lần chạy chúng trông như "chưa huấn luyện".

**Kiểm chứng sau khi sửa**: accuracy do `run_all_viz.py` tính nay **trùng khớp
từng chữ số** với `test_all.py` tính độc lập — RF 0,9451 · XGBoost 0,9949 ·
LightGBM 0,9963. Hai đường mã khác nhau cho cùng kết quả là bằng chứng phép sửa
đúng, không phải chỉ hết báo lỗi.

---

## 11. Những gì chưa kiểm chứng được

| Việc | Vì sao |
|---|---|
| Lệnh cho RADAR và UOS | không còn dữ liệu CSV thô trong phiên này; mã đường dẫn đã rà soát tĩnh nhưng **chưa chạy** |
| `--stage scratch` của GAT-Transformer | thí nghiệm quyết định, vẫn chưa chạy lần nào |
| ≥5 hạt giống cho mọi cấu hình chính | vẫn là thiếu sót giá trị nhất |
| Ablation `--no-edge` với `allow_abbrev=False` | chưa chạy lại sau khi vá |
| `delayed_reply` là gì | đã tra 4 nguồn, không xác định được; cần xem tên thư mục bộ RADAR trên máy |
