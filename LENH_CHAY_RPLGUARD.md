# Lệnh chạy — bộ RPLGuard và trực quan hoá cho cả 4 bộ

Phiên bản framework **v3.41**: thêm bộ RPLGuard, t-SNE từng tầng cho NE-GAT,
và trình chạy t-SNE + ma trận nhầm lẫn cho mọi mô hình trên cả bốn bộ.

---

## 1. Bộ RPLGuard — ba điều phải biết trước khi chạy

### 1.1 Nhãn nằm ở cột `node_type`

Tệp readme đi kèm bộ dữ liệu **không nói nhãn nằm ở đâu**; nó chỉ ghi rằng
người dùng phải tự làm bước *"label assignment"*. Tôi đã dò bằng dữ liệu:

| Cột | Giá trị quan sát được | Dùng làm nhãn? |
|---|---|---|
| `node_type` | `CHILD_NODE`, `ROOT_NODE`, `EDGE_NODE`, `Blackhole Attack`, `Greyhole Attack` | **Có** |
| `node_honest_level` | 768–790 cho **cả hai** nhãn | Không |

Phiên bản builder cũ trong framework giả định nhãn nằm ở `node_honest_level`
— **sai**, đã viết lại.

### 1.2 Cột `To` dùng ký hiệu Cooja, không phải số

Giá trị cột `To` có dạng `[14 d]` (nút 14, truyền trực tiếp). Nếu ép thẳng
bằng `to_numeric` thì **93,9% số dòng bị loại âm thầm**:

| | Dòng sinh được cạnh | Tổng cạnh dựng được |
|---|---|---|
| Ép trực tiếp (sai) | 1.215/19.939 = **6,1%** | 3.409 |
| Trích bằng regex (đúng) | 19.939/19.939 = **100%** | **13.473** |

Nếu không sửa, đồ thị gần như không còn cạnh và toàn bộ ý tưởng "đặc trưng
cạnh trong hệ số chú ý" mất ý nghĩa trên bộ này.

### 1.3 Rò rỉ nhãn — nghiêm trọng, phải xử lý

Hai nguồn rò rỉ, đều đã đo:

**(a) `node_id` rò rỉ hoàn toàn.** Nút tấn công luôn là **7, 17, 27** ở mọi
tệp, và mang nhãn tấn công **100% thời gian** (không có giai đoạn sạch trước
khi tấn công bắt đầu, khác hẳn RADAR có 500 s đầu sạch và UOS có 100 s đầu
sạch). Luật `node_id ∈ {7,17,27}` cho **F1 = 1,0000 trên cả 6 tệp**.
→ Builder **loại bỏ** `node_id`, `From`, `To` và mọi cột địa chỉ khỏi vector
đặc trưng. (Địa chỉ nhúng thẳng số hiệu nút: nút 23 → `fe80::212:7423232323`.)

**(b) `rank` rò rỉ mạnh.** Giá trị **257** xuất hiện ở **94,6%** dòng tấn công
và **0%** dòng bình thường (nút thường dùng 512/768/1024).

Đo trên đồ thị đã dựng (cửa sổ 5 s, 8.069 nút, tấn công 11,0%):

| `rank_mode` | F1 gốc quyết định trên rank | F1 cây độ sâu 6 (mọi đặc trưng) |
|---|---|---|
| `raw` | 0,8542 | 0,9375 |
| `relative` | 0,8269 | 0,9375 |
| **`drop`** (mặc định) | — | **0,5387** |

Phép chuẩn hoá tương đối **không đủ** — chỉ hạ F1 từ 0,854 xuống 0,827. Chỉ
khi bỏ hẳn rank thì trần ghi nhớ mới tụt xuống 0,5387.

> **Bỏ rank KHÔNG phải là mất mát tín hiệu.** Hai tấn công của bộ này là
> Blackhole và Greyhole — tấn công **vứt gói**, không phải tấn công thao túng
> rank. Một nút blackhole không cần giả mạo rank để vứt gói. Giá trị 257 vì
> vậy là **hiện vật của cấu hình mô phỏng**, không phải chữ ký nhân quả của
> hành vi tấn công. Tín hiệu nhân quả thật nằm ở `drop_count`, `forward_pkts`
> và `etx` — và builder giữ nguyên ba đặc trưng đó.

### 1.4 Một bất thường trong chính bộ dữ liệu

Ba tệp Greyhole đặt tên `10nodes`, `20nodes`, `30nodes` nhưng **cả ba đều là
mạng 30 nút với 3 kẻ tấn công** (node_id chạy 1–30, kẻ tấn công 7/17/27).
Chỉ ba tệp Blackhole là khớp tên (10 nút/1 kẻ, 20 nút/2 kẻ, 30 nút/3 kẻ),
đúng như readme mô tả. Nên khi viết luận văn, **không được nói bộ RPLGuard
có ba bậc mạng cho cả hai loại tấn công** — điều đó chỉ đúng với Blackhole.

---

## 2. Chuyển đổi dữ liệu RPLGuard

```bash
# Giải nén bộ dữ liệu sao cho có cấu trúc:
#   data/rplguard_raw/Blackholeattack-DatasetFiles/*.csv
#   data/rplguard_raw/Greyholeattack-DatasetFiles/*.csv

python scripts/convert_data.py --dataset rplguard --data-dir data/rplguard_raw --window 5 --split chrono --out-dir data/rplguard_chrono
```

**Dùng `--split chrono`, KHÔNG dùng `--split-by-file`.** Lý do đo được:

| Cách chia | train / val / test (đồ thị) | Vấn đề |
|---|---|---|
| `--split chrono` (7:1:2 trong từng tệp) | 301 / 42 / 90 | đủ 3 lớp ở cả 3 tập; δ̄ = 15,500 |
| `--split-by-file 0,1,3,4:2:5` | 282 / 60 / 91 | **val có 0 nút greyhole, test có 0 nút blackhole** |

Bộ chỉ có 6 tệp, mà 3 tệp đầu là blackhole và 3 tệp sau là greyhole, nên mọi
cách chia theo tệp đều làm mất một lớp khỏi tập kiểm thử. Khi đó macro-F1
3 lớp không còn ý nghĩa.

**Biến thể nhị phân** (gộp blackhole + greyhole thành một lớp "attack"):

```bash
python scripts/convert_data.py --dataset rplguard_binary \
       --data-dir data/rplguard_raw --window 5 \
       --split chrono --out-dir data/rplguard_bin
```

**Chạy đối chứng để tái hiện rò rỉ** (chỉ để chứng minh trong luận văn, không
dùng làm kết quả chính):

```bash
RPLGUARD_RANK_MODE=raw python scripts/convert_data.py --dataset rplguard \
       --data-dir data/rplguard_raw --window 5 \
       --split chrono --out-dir data/rplguard_leaky
```

Script sẽ in cảnh báo và kiểm toán rò rỉ sẽ cho thấy chênh lệch.

---

## 3. Huấn luyện trên RPLGuard

Giống hệt ba bộ kia, chỉ đổi `--data-dir`:

```bash
# toàn bộ baseline + hai mô hình đề xuất
python scripts/train_all.py --data-dir data/rplguard_chrono \
       --ckpt-dir checkpoints_rplguard

# ATA
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat \
       --K 3 --lambda-d 0.1 --epochs 120 --ckpt-dir ck_ata_rplguard

# GAT-Transformer (bộ này ít đồ thị thuần bình thường -> cân nhắc scratch)
python scripts/train_ssl.py --data-dir data/rplguard_chrono --stage scratch \
       --backbone negat --epochs 120 --ckpt-dir ck_gtids_rplguard \
       --d-model 32 --dim-ff 64 --tr-layers 1
```

> **Lưu ý về quy mô.** RPLGuard chỉ sinh được **433 đồ thị** từ 6 tệp, so với
> 26.100 của RADAR. Tập kiểm thử chỉ có **1.367 nút**, trong đó 39 blackhole
> và 109 greyhole. Với cỡ mẫu này, chênh lệch macro-F1 dưới khoảng 0,03 giữa
> các mô hình **không phân biệt được với nhiễu**. Phải chạy nhiều hạt giống và
> báo cáo độ lệch chuẩn, nếu không thì chỉ nên dùng bộ này như một phép
> **kiểm tra tính khả chuyển**, không phải bảng xếp hạng.

---

## 4. t-SNE và ma trận nhầm lẫn cho MỌI mô hình trên CẢ 4 BỘ

### 4.1 Chạy một bộ

```bash
python scripts/run_all_viz.py --data-dir data/rplguard_chrono --ckpt-dir checkpoints_rplguard --gt-dir ck_gtids_rplguard --out-dir viz/rplguard --tag rplguard
```
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir checkpoints_rplguard

```bash
python scripts/run_all_viz.py --data-dir data/radar_chrono --ckpt-dir checkpoints_radar --gt-dir ck_gtids_radar --out-dir viz/radar --tag radar
```
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir checkpoints_radar

```bash
python scripts/run_all_viz.py --data-dir data/rplbeh_chrono --ckpt-dir checkpoints_rplbeh --gt-dir ck_gtids_rplbeh --out-dir viz/rplbeh --tag rplbeh
```
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir checkpoints_rplbeh

### 4.2 Chạy cả bốn bộ

```bash
bash scripts/run_all_viz_4ds.sh
```

Sửa các biến đầu tệp cho khớp đường dẫn trên máy, hoặc truyền qua biến môi
trường:

```bash
DATA_RADAR=data/radar_chrono CKPT_RADAR=checkpoints_radar \
DATA_GUARD=data/rplguard_chrono CKPT_GUARD=checkpoints_rplguard \
bash scripts/run_all_viz_4ds.sh
```

### 4.3 Quy tắc t-SNE đã hiện thực

| Mô hình | t-SNE | Số bảng |
|---|---|---|
| **EdgeGAT-IDS** | **từng tầng** | `tầng 1 … tầng K` + `trước bộ phân loại` |
| **NE-GAT-IDS** | **từng tầng** | `tầng 1 … tầng K` + `trước bộ phân loại` |
| GAT-Transformer | tầng cuối | 1 |
| MLP, GCN, E-GraphSAGE | tầng cuối | 1 |
| RF, XGBoost, LightGBM | tầng cuối | 1 |

> **Một thay đổi trong mã mô hình đã phải thực hiện.** `NEGAT.forward` trước
> đây chỉ trả `{final, logits}` khi gọi `return_embeddings=True`, nên **không
> vẽ được t-SNE từng tầng cho NE-GAT**. Đã bổ sung thu thập biểu diễn sau mỗi
> tầng, dùng đúng quy ước khoá `h1..hK` của EdgeAwareGAT. Đã kiểm chứng rằng
> đường chạy bình thường (`forward` thường và `return_feat=True`) cho kết quả
> **y hệt trước khi sửa**.

> **Về mô hình cây.** RF, XGBoost và LightGBM **không có tầng ẩn**. Thay vì bỏ
> qua, script dùng **véc-tơ xác suất lớp** làm "biểu diễn cuối" — đó đúng là
> thứ cây sinh ra ngay trước phép `argmax`. Mỗi hình đều ghi chú rõ điều này
> để không ai hiểu nhầm rằng cây có không gian nhúng như mạng nơ-ron.

### 4.4 Tệp sinh ra

```
viz/<bộ>/
  cm_<bộ>_<mô hình>.png      ma trận nhầm lẫn (chuẩn hoá theo hàng)
  cm_<bộ>_<mô hình>.csv      ma trận nhầm lẫn dạng số nguyên
  tsne_<bộ>_<mô hình>.png    t-SNE, có điểm silhouette trên mỗi bảng
  viz_summary_<bộ>.json      accuracy và số bảng t-SNE của từng mô hình
```

---

## 5. Số liệu thật của bốn bộ — để đối chiếu

| | RADAR | UOS | RPL-Beh | RPLGuard |
|---|---|---|---|---|
| Tệp / kịch bản | 87 (0 bỏ) | 77 tìm thấy, **76 dùng được** | 95 mô phỏng | 6 |
| Cửa sổ | 5 s | 5 s | **480 s** (bộ ấn định) | 5 s |
| Đồ thị | **26.100** | 3.848 | 7.030 | **433** |
| Đồ thị / tệp (TB) | 300,0 | 50,6 | 74,0 | 72,2 |
| `d_n` / `d_e` | 29 / 10 | 13 / 10 | 24 / 10 | 13 / 9 |
| Số lớp | 16 | 2 | 5 | 3 |
| train / val / test | 18.270 / 2.610 / 5.220 | 2.661 / 348 / 839 | 4.958 / 1.110 / 962 | 301 / 42 / 90 |
| Nút tập test | 98.579 | 8.904 | 18.177 | 1.367 |
| δ̄ (cửa sổ) | 60,500 | 14,969 | — | 15,500 |

Số đồ thị mỗi tệp của RPLGuard (cửa sổ 5 s):

| Tệp | Dòng | Đồ thị |
|---|---|---|
| Blackhole 10 nút | 1.565 | 58 |
| Blackhole 20 nút | 4.132 | 60 |
| Blackhole 30 nút | 12.778 | 60 |
| Greyhole (tên 10 nút) | 19.337 | 82 |
| Greyhole (tên 20 nút) | 18.717 | 82 |
| Greyhole (tên 30 nút) | 19.939 | 91 |
| **Tổng** | **76.468** | **433** |
