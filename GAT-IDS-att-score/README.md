# GAT-IDS · NE-GAT-IDS

**Phát hiện xâm nhập trong mạng IoT định tuyến RPL bằng mạng nơ-ron đồ thị có chú ý, huấn luyện với căn chỉnh miền thời gian thích nghi (ATA).**


---

## Ý tưởng cốt lõi

Tấn công RPL mang **bản chất quan hệ**. Một nút công bố `rank = 128` không có gì bất thường nếu xét riêng lẻ — giá trị đó hoàn toàn hợp lệ với một nút gần gốc. Nó chỉ trở thành bằng chứng khi đặt cạnh quan sát của các nút láng giềng:

```
Δᵢ = | rank nút TỰ KHAI  −  rank trung bình LÁNG GIỀNG ghi nhận |
```

Đại lượng `Δᵢ` **không tồn tại** trong biểu diễn dữ liệu dạng bảng, vì tính nó đòi hỏi biết nút nào là láng giềng của nút nào. Đây là lý do nền tảng dẫn tới lựa chọn mô hình đồ thị.

Khung này mô hình hoá mỗi cửa sổ thời gian của mạng thành một đồ thị DODAG: **nút** mang đặc trưng hành vi thiết bị, **cạnh** mang đặc trưng chất lượng liên kết. Bài toán được đặt ở mức **phân loại nút**, để hệ thống không chỉ trả lời *"mạng có bị tấn công không"* mà chỉ đích danh **thiết bị nào**.

---

## Hai kiến trúc

### GAT-IDS

Ba tầng GATv2 với đặc trưng cạnh đưa **trực tiếp** vào hệ số chú ý:

```
z_ij = aᵀ · LeakyReLU( W_l·h_i + W_r·h_j + W_e·e_ij )
```

Điểm mấu chốt: `e_ij` tham gia vào việc **tính** hệ số chú ý, chứ không chỉ được cộng vào thông điệp sau khi hệ số đã cố định. Nhờ đó mô hình học được các quy tắc như *hạ trọng số láng giềng nối qua liên kết có tỉ lệ xung đột cao*, hoặc *phân biệt liên kết cha–con thật với liên kết chỉ nghe thấy nhau*.

103.956 tham số trên bộ RADAR.

### NE-GAT-IDS

Kiến trúc **hai kênh** tách biệt, hợp nhất trước bộ phân loại:

- **Kênh nút** — GATv2 không dùng đặc trưng cạnh, trả lời *"láng giềng nào quan trọng?"*
- **Kênh cạnh** — chú ý riêng trên liên kết, trả lời *"liên kết nào quan trọng?"*

Kênh cạnh có một chi tiết đáng lưu ý. Phép tổng hợp có trọng số softmax là **tổ hợp lồi**, nên **bất biến với số lượng liên kết**: nút có 3 liên kết bất thường và nút có 30 liên kết bất thường cho cùng một véc-tơ. Chứng minh một dòng — nếu nhân đôi mọi láng giềng thì mẫu số softmax nhân đôi nên `β'ᵢⱼ = ½·βᵢⱼ`, tổng có `2m` số hạng mỗi số hạng bằng nửa giá trị cũ, tổng không đổi. Kết quả này nhất quán với định lý của Xu và cộng sự (ICLR 2019). Để khôi phục, kênh cạnh nối thêm một thành phần mang thông tin bậc theo dạng degree scaler của Corso và cộng sự (NeurIPS 2020).

62.224 tham số — **ít hơn GAT-IDS khoảng 41%**.

---

## Adaptive Temporal Alignment

ATA coi tập huấn luyện là tập hợp **có thứ tự** gồm `K` miền thời gian liền kề, thay vì một kho mẫu đồng nhất:

```
L_ATA = (1/K)·Σₖ L_cls⁽ᵏ⁾  +  (λ_d/|P|)·Σ₍ₐ,ᵦ₎ L_trans⁽ᵃᵇ⁾
```

Số hạng thứ nhất giữ cho bộ mã hoá còn phân biệt được **trong từng** giai đoạn, nên việc căn chỉnh không thể đạt được bằng cách làm sụp cấu trúc lớp. Số hạng thứ hai kéo phân bố đặc trưng của các giai đoạn về một hình học chung.

**Khác biệt với DANN và CDAN:** các phương pháp thích ứng miền cổ điển là nguồn → đích và cần mẫu của miền **đích** khi huấn luyện. Ở đây miền đích là lưu lượng **tương lai** — chưa xảy ra nên không lấy mẫu được. ATA ép tính nhất quán **lẫn nhau** giữa các giai đoạn quá khứ đã quan sát.

ATA **không đụng tới kiến trúc**. Nó chỉ thay hàm mục tiêu và cách lấy mẫu, nên cùng một mã huấn luyện dùng được cho mọi kiến trúc và chênh lệch đo được quy đúng về ATA.

> **Sai lệch có chủ ý so với công thức gốc.** Bài báo gốc viết `Σₖ L_cls⁽ᵏ⁾` dạng **tổng**. Khi so cấu hình ATA (K=3) với đối chứng (K=1), dạng tổng làm gradient lớn gấp K lần, tức đổi ngầm tốc độ học và tạo yếu tố nhiễu cho phép so sánh. Mặc định ở đây là dạng **trung bình**; dùng `--cls-reduction sum` để tái lập đúng công thức gốc.

Tham khảo: Phan, Dang Le, Vu, Vo (2026). *Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift.*

---

## Giao thức đánh giá chống rò rỉ

Đây là phần được đầu tư nhiều nhất trong khung này, vì nó quyết định mọi con số sau đó có ý nghĩa hay không.

### Ba dạng rò rỉ và cơ chế phòng chống

| Dạng | Biểu hiện | Cơ chế |
|:---|:---|:---|
| **Thời gian** | Các cửa sổ liền kề của cùng một thiết bị nằm ở cả train lẫn test | Chia theo thứ tự thời gian trong từng tệp |
| **Đặc trưng** | Một cột thực chất là bản sao trá hình của nhãn | Kiểm toán rò rỉ nhãn hai tiêu chí |
| **Chuẩn hoá** | Tham số chuẩn hoá ước lượng trên toàn bộ dữ liệu | Ước lượng λ, μ, σ **chỉ** trên tập huấn luyện |

### Chẩn đoán rò rỉ thời gian

```
δ(i) = min_{j ∈ train} |τᵢ − τⱼ|          δ̄ = trung bình δ(i) trên tập test
```

Toán tử `min` là cố ý: rò rỉ mang tính **cực trị**, chỉ cần một mẫu huấn luyện liền kề là đủ để mô hình nội suy. Nếu lấy trung bình khoảng cách thay vì cực tiểu, con số sẽ bị các mẫu ở xa kéo lên và che mất rò rỉ cục bộ.

Đo được trên bộ RADAR, mỗi tệp 300 cửa sổ:

| Chế độ chia | δ̄ (cửa sổ) | Quy ra giây |
|:---|---:|---:|
| Ngẫu nhiên | **1,098** | 5,5 s |
| Theo thời gian | **60,5** | 302 s |

Chênh **55 lần**. Giá trị 60,5 khớp chính xác với giá trị giải tích `(31+90)/2`, xác nhận cài đặt đúng như thiết kế.

> **Cạm bẫy về đơn vị.** δ̄ của ba bộ dữ liệu **không so trực tiếp được với nhau**, vì `win_idx` có đơn vị khác nhau: RADAR và UOS đếm theo cửa sổ 5 giây, RPL-Beh theo chỉ số phút. Quy về giây: 302 s, 75 s và 7.440 s. δ̄ cũng tỉ lệ thuận với số cửa sổ mỗi tệp theo `δ̄ ≈ 0,2·n`, nên nó là chỉ số **tương đối**, không phải ngưỡng tuyệt đối.

### Kiểm toán rò rỉ nhãn

Hai tiêu chí, chạy tự động trước khi huấn luyện:

```
nMI(j) = I( bin(x_j) ; y ) / H(y)              trần = H(1[y>0]) / H(y)
cột j bị gắn cờ  ⟺  F1( cây_độ_sâu_2(x_j), 1[y>0] ) > 0,90
```

Ngưỡng nMI phải là **ngưỡng động**: một oracle nhị phân hoàn hảo về lý thuyết không thể đạt nMI gần 1 trong bài toán nhiều lớp vì nó không cho biết loại tấn công nào. Với bộ IoT-RPL, `H(5 lớp) = 0,933` và `H(nhị phân) = 0,573` cho trần `0,614` — một ngưỡng cố định 0,85 sẽ **không bao giờ** kích hoạt.

Tiêu chí quyết định là phép thử gốc-cây. Cây độ sâu 2 chỉ tạo được bốn lá; nếu bốn khoảng đó đã đủ tái tạo nhãn với F1 vượt 0,90 trên hàng chục nghìn mẫu thì cột đó gần như chắc chắn là bản sao của nhãn.

---

## Cài đặt

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4.

---

## Bắt đầu nhanh

```bash
# Bước 1 — chuyển sang đồ thị, chia theo thứ tự thời gian, kiểm toán rò rỉ
python scripts/convert_data.py --dataset radar --data-dir data/radar \
       --split chrono --alpha 1.0 --out-dir data/radar_chrono

# Bước 2 — huấn luyện với ATA
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 3 --lambda-d 0.5 --disc mmd --epochs 120 --ckpt-dir ck_negat_ata

# Bước 3 — đánh giá đa lớp và nhị phân
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_negat_ata --binary
```

# LỆNH CHẠY TOÀN BỘ PIPELINE — GAT-IDS · NE-GAT-IDS · ATA

Phiên bản v3.33. Mọi lệnh chạy từ thư mục gốc framework.

---

## 0. Điều kiện tiên quyết

```bash
pip install torch torch_geometric xgboost lightgbm scikit-learn pandas matplotlib
```

Cấu trúc thư mục dữ liệu:

```
data/radar/     <- các thư mục kịch bản tấn công, mỗi thư mục chứa *.csv
data/uos/       <- các tệp log của bộ UOS_IOTSH_2024
data/rplbeh/    <- RPL-IDS-Beh.csv  (tải từ kho HUNSR/RPL-IDS-Behavior-Dataset)
```

---

## 1. BƯỚC 1 — Chuyển dữ liệu sang đồ thị, chia theo THỨ TỰ THỜI GIAN

Ba bộ dữ liệu dùng **cùng một chiến lược chia** để so sánh được với nhau.

```bash
# ── RADAR (16 lớp) ──
python scripts/convert_data.py --dataset radar --data-dir data/radar --split chrono --alpha 1.0 --out-dir data/radar_chrono

# ── UOS (2 lớp) ──
python scripts/convert_data.py --dataset uos --data-dir data/uos --split chrono --alpha 1.0 --out-dir data/uos_chrono

# ── RPL-Beh (5 lớp) ──
python scripts/convert_data.py --dataset rplbeh --data-dir data/rplbeh --split chrono --alpha 1.0 --out-dir data/rplbeh_chrono
```

**Ba điều cần kiểm ngay trong log:**

1. `Chẩn đoán rò rỉ thời gian: δ̄ = …` — giá trị phải **lớn hơn 2** rõ rệt. Nếu gần 0 thì phép chia vẫn đang là ngẫu nhiên và mọi kết quả ATA sau đó sẽ không đo được thứ cần đo.
2. `Không phát hiện rò rỉ` hoặc `WARNING — Đặc trưng RÒ RỈ` — nếu có cảnh báo, phải xử lý trước khi huấn luyện.
3. Tỉ lệ đồ thị train/val/test **không** khớp chính xác 7:1:2 là bình thường, vì mỗi tệp sinh số cửa sổ khác nhau.

### 1b. Quét tỉ lệ giữ lại (tách ngân sách dữ liệu khỏi dịch chuyển thời gian)

```bash
for A in 0.5 0.6 0.8 1.0; do
  python scripts/convert_data.py --dataset radar --data-dir data/radar \
         --split chrono --alpha $A --out-dir data/radar_a$A
done
```

Nếu hiệu năng tăng **đơn điệu** theo α thì kích thước mẫu là nút thắt; nếu **đi ngang hoặc dao động** thì dịch chuyển thời gian mới là nguyên nhân chính.

### 1c. Chế độ đối chứng để tái lập rò rỉ

```bash
python scripts/convert_data.py --dataset radar --data-dir data/radar \
       --split random --out-dir data/radar_random
```

So hai giá trị δ̄ giữa `chrono` và `random` chính là bằng chứng định lượng cho khoảng trống nghiên cứu thứ hai.

---

## 2. BƯỚC 2 — Huấn luyện

### 2a. Baseline dạng CÂY — không dùng được ATA

ATA là kỹ thuật dựa trên **hàm mất mát**, nên chỉ áp dụng được cho mô hình huấn luyện bằng hạ gradient. Rừng ngẫu nhiên, XGBoost và LightGBM **không** dùng được ATA; chúng vẫn được huấn luyện bằng `train_all.py` trên **cùng** tập chia theo thời gian, nên vẫn so sánh được.

```bash
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_radar
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --seed 42 --ckpt-dir ck_trees_radar_seed42
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --seed 43 --ckpt-dir ck_trees_radar_seed43
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --seed 44 --ckpt-dir ck_trees_radar_seed44
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --seed 45 --ckpt-dir ck_trees_radar_seed45
python scripts/train_all.py --data-dir data/radar_chrono --models RF,XGBoost,LightGBM --resample off --seed 46 --ckpt-dir ck_trees_radar_seed46
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_trees_radar_seed42 --binary --out kq_trees_radar_seed42.json
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_trees_radar_seed43 --binary --out kq_trees_radar_seed43.json
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_trees_radar_seed44 --binary --out kq_trees_radar_seed44.json
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_trees_radar_seed45 --binary --out kq_trees_radar_seed45.json
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_trees_radar_seed46 --binary --out kq_trees_radar_seed46.json


python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_uos
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --seed 42 --ckpt-dir ck_trees_uos_seed42
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --seed 43 --ckpt-dir ck_trees_uos_seed43
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --seed 44 --ckpt-dir ck_trees_uos_seed44
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --seed 45 --ckpt-dir ck_trees_uos_seed45
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --seed 46 --ckpt-dir ck_trees_uos_seed46


python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_rplbeh
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --seed 42 --ckpt-dir ck_trees_rplbeh_seed42
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --seed 43 --ckpt-dir ck_trees_rplbeh_seed43
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --seed 44 --ckpt-dir ck_trees_rplbeh_seed44
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --seed 45 --ckpt-dir ck_trees_rplbeh_seed45
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --seed 46 --ckpt-dir ck_trees_rplbeh_seed46
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir ck_trees_rplbeh_seed42 --binary --out kq_trees_rplbeh_seed42.json
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir ck_trees_rplbeh_seed43 --binary --out kq_trees_rplbeh_seed43.json
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir ck_trees_rplbeh_seed44 --binary --out kq_trees_rplbeh_seed44.json
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir ck_trees_rplbeh_seed45 --binary --out kq_trees_rplbeh_seed45.json
python scripts/test_all.py --data-dir data/rplbeh_chrono --ckpt-dir ck_trees_rplbeh_seed46 --binary --out kq_trees_rplbeh_seed46.json

python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_rplguard
python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --seed 42 --ckpt-dir ck_trees_rplguard_seed42
python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --seed 43 --ckpt-dir ck_trees_rplguard_seed43
python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --seed 44 --ckpt-dir ck_trees_rplguard_seed44
python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --seed 45 --ckpt-dir ck_trees_rplguard_seed45
python scripts/train_all.py --data-dir data/rplguard_chrono --models RF,XGBoost,LightGBM --resample off --seed 46 --ckpt-dir ck_trees_rplguard_seed46
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir ck_trees_rplguard_seed42 --binary --out kq_trees_rplguard_seed42.json
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir ck_trees_rplguard_seed43 --binary --out kq_trees_rplguard_seed43.json
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir ck_trees_rplguard_seed44 --binary --out kq_trees_rplguard_seed44.json
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir ck_trees_rplguard_seed45 --binary --out kq_trees_rplguard_seed45.json
python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir ck_trees_rplguard_seed46 --binary --out kq_trees_rplguard_seed46.json
```

### 2b. Baseline mạng nơ-ron — CÓ ATA

```bash
for A in mlp gcn egraphsage; do
  python scripts/train_ata.py --data-dir data/radar_chrono --arch $A --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_${A}_ata
done
```

#--> For RADAR DATASET
```
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_egraphsage_ata_radar
```
Baseline			≈ 62k (bằng NE-GAT)					≈ 104k (bằng GAT-IDS)
E-GraphSAGE			--hidden-dim 184 → 60.882			--hidden-dim 256 → 102.354
MLP					--hidden-dim 216 → 61.755			--hidden-dim 288 → 102.867
GCN					--hidden-dim 960 → 62.066			không đạt được

python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_radar
[ATA with no shuffle in batch(k)] python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_ata_noshuffle_radar
[NO ATA] python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_noata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_mlp_ata_radar_seed42
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_mlp_ata_radar_seed43
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_mlp_ata_radar_seed44
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_mlp_ata_radar_seed45
python scripts/train_ata.py --data-dir data/radar_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_mlp_ata_radar_seed46

python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_radar
[ATA with no shuffle in batch(k)] python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_ata_noshuffle_radar
[NO ATA] python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_noata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_gcn_ata_radar_seed42
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_gcn_ata_radar_seed43
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_gcn_ata_radar_seed44
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_gcn_ata_radar_seed45
python scripts/train_ata.py --data-dir data/radar_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_gcn_ata_radar_seed46

python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_esage_radar
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_esage_noshuffle_radar
[NO ATA] python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_egraphsage_noata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_esage_radar_seed42
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_esage_radar_seed43
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_esage_radar_seed44
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_esage_radar_seed45
python scripts/train_ata.py --data-dir data/radar_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_esage_radar_seed46





#--> For UOS DATASET
```
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_egraphsage_ata_uos
```
Baseline			≈ 62k (bằng NE-GAT)					≈ 104k (bằng GAT-IDS)
E-GraphSAGE			--hidden-dim 184 → 60.882			--hidden-dim 256 → 102.354
MLP					--hidden-dim 216 → 61.755			--hidden-dim 288 → 102.867
GCN					--hidden-dim 960 → 62.066			không đạt được

python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_mlp_ata_uos_seed42
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_mlp_ata_uos_seed43
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_mlp_ata_uos_seed44
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_mlp_ata_uos_seed45
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_mlp_ata_uos_seed46



python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_gcn_ata_uos_seed42
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_gcn_ata_uos_seed43
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_gcn_ata_uos_seed44
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_gcn_ata_uos_seed45
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_gcn_ata_uos_seed46


python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_esage_uos_seed42
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_esage_uos_seed43
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_esage_uos_seed44
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_esage_uos_seed45
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_esage_uos_seed46






#--> For RPL-Beh DATASET
```
python scripts/train_ata.py --data-dir data/uos_chrono --arch mlp --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplbeh
python scripts/train_ata.py --data-dir data/uos_chrono --arch gcn --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplbeh
python scripts/train_ata.py --data-dir data/uos_chrono --arch egraphsage --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_egraphsage_ata_rplbeh
```

Baseline			≈ 62k (bằng NE-GAT)					≈ 104k (bằng GAT-IDS)
E-GraphSAGE			--hidden-dim 184 → 60.882			--hidden-dim 256 → 102.354
MLP					--hidden-dim 216 → 61.755			--hidden-dim 288 → 102.867
GCN					--hidden-dim 960 → 62.066			không đạt được

python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 288 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 256 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_esage_rplbeh

python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplbeh
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_ata_noshuffle_rplbeh
[NO ATA] python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_noata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_mlp_ata_rplbeh_seed42
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_mlp_ata_rplbeh_seed43
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_mlp_ata_rplbeh_seed44
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_mlp_ata_rplbeh_seed45
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_mlp_ata_rplbeh_seed46

python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplbeh
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_ata_noshuffle_rplbeh
[NO ATA] python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_noata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_gcn_ata_rplbeh_seed42
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_gcn_ata_rplbeh_seed43
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_gcn_ata_rplbeh_seed44
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_gcn_ata_rplbeh_seed45
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_gcn_ata_rplbeh_seed46

python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_esage_rplbeh
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_esage_noshuffle_rplbeh
[NO ATA] python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_egraphsage_noata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_esage_rplbeh_seed42
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_esage_rplbeh_seed43
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_esage_rplbeh_seed44
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_esage_rplbeh_seed45
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_esage_rplbeh_seed46

#--> For RPLGuard DATASET
```
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_egraphsage_ata_rplguard
```

Baseline			≈ 62k (bằng NE-GAT)					≈ 104k (bằng GAT-IDS)
E-GraphSAGE			--hidden-dim 184 → 60.882			--hidden-dim 256 → 102.354
MLP					--hidden-dim 216 → 61.755			--hidden-dim 288 → 102.867
GCN					--hidden-dim 960 → 62.066			không đạt được



python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_esage_rplguard

python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_mlp_ata_rplguard
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_ata_noshuffle_rplguard
[NO ATA] python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_mlp_noata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_mlp_ata_rplguard_seed42
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_mlp_ata_rplguard_seed43
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_mlp_ata_rplguard_seed44
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_mlp_ata_rplguard_seed45
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch mlp --hidden-dim 216 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_mlp_ata_rplguard_seed46

python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gcn_ata_rplguard
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_ata_noshuffle_rplguard
[NO ATA] python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gcn_noata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_gcn_ata_rplguard_seed42
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_gcn_ata_rplguard_seed43
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_gcn_ata_rplguard_seed44
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_gcn_ata_rplguard_seed45
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch gcn --hidden-dim 960 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_gcn_ata_rplguard_seed46

python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_esage_rplguard
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_esage_noshuffle_rplguard
[NO ATA] python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_egraphsage_noata_rplguard
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 42 --ckpt-dir ck_esage_rplguard_seed42
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 43 --ckpt-dir ck_esage_rplguard_seed43
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 44 --ckpt-dir ck_esage_rplguard_seed44
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 45 --ckpt-dir ck_esage_rplguard_seed45
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch egraphsage --hidden-dim 184 --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --seed 46 --ckpt-dir ck_esage_rplguard_seed46



### 2c. Hai mô hình chính

#--> For RADAR DATASET
```bash
# GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_ata_radar
[NO ATA] python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gat_noata_radar

# NE-GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_ata_radar
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_ata_noshuffle_radar
[NO ATA] python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_noata_radar
```

#--> For UOS DATASET
```bash
# GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/uos_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_ata_uos
[NO ATA]: python scripts/train_ata.py --data-dir data/uos_chrono --arch edgegat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gat_noata_uos

# NE-GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_ata_uos
[NO ATA]: python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_noata_uos
```

#--> For RPL-BEH DATASET
```bash
# GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_ata_rplbeh
[NO ATA]: python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch edgegat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gat_noata_rplbeh

# NE-GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_ata_noshuffle_rplbeh
[NO ATA]: python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_noata_rplbeh

```

#--> For RPLGUARD DATASET
```bash
# GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_ata_rplguard
[NO ATA]: python scripts/train_ata.py --data-dir data/rplguard_chrono --arch edgegat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_gat_noata_rplguard

# NE-GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_ata_rplguard
[ATA with no shuffle in batch(k)]python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_ata_noshuffle_rplguard
[NO ATA]: python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 1 --lambda-d 0 --disc mmd --domain-strategy quantile --epochs 120 --no-shuffle --ckpt-dir ck_negat_noata_rplguard

```


### 2d. Nhánh đối chứng KHÔNG có ATA

```bash
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 1 --lambda-d 0 --epochs 120 --ckpt-dir ck_gat_noata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 1 --lambda-d 0 --epochs 120 --ckpt-dir ck_negat_noata_radar
```

Đặt `--K 1 --lambda-d 0` là **có chủ ý**: cấu hình này thoái hoá về cực tiểu hoá rủi ro thực nghiệm thông thường nhưng **vẫn đi qua đúng một đường mã**, nên loại được mọi khác biệt cài đặt khỏi phép so sánh.

---

## 3. Ablation đặc trưng cạnh — hai câu hỏi khác nhau

Anh hỏi hai điều khác nhau, và chúng cần hai cặp lệnh khác nhau.

### 3a. "Có đặc trưng cạnh" so với "không có đặc trưng cạnh"

Áp cho **GAT-IDS**. Khác biệt duy nhất là cờ `--no-edge`:

```bash
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_edge_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --no-edge --ckpt-dir ck_gat_noedge_radar
```

*Lưu ý:* cờ `--no-edge` hiện chỉ có trong `train_all.py`. Nếu `train_ata.py` báo lỗi không nhận cờ, hãy đặt `use_edge_features: false` trong `configs/default.yaml` trước khi chạy nhánh thứ hai, rồi đặt lại `true`.

### 3b. "Cạnh CÓ trong attention" so với "cạnh KHÔNG trong attention"

Đây là câu hỏi khác và chỉ áp được cho **NE-GAT-IDS**, vì kiến trúc này tách riêng kênh cạnh:

```bash
# cạnh tham gia điểm chú ý cùng ngữ cảnh hai đầu mút (mặc định)
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --edge-score full --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_full

# điểm chú ý CHỈ tính từ trạng thái cạnh, cô lập khỏi biểu diễn nút
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --edge-score edge_only --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_edgeonly
```

Cặp này là **phép thử trực tiếp** cho giả thuyết rằng tín hiệu cạnh bị lấn át khi trộn chung với đặc trưng nút. Nếu `edge_only` không tốt hơn `full`, giả thuyết không được ủng hộ và phần lập luận phải viết lại.

### 3c. Hai ablation riêng của NE-GAT

```bash
# mất thông tin số lượng liên kết (tái lập thiết kế gốc)
python scripts/train_ata.py --arch negat --edge-agg mean ... --ckpt-dir ck_negat_mean
# tắt cập nhật trạng thái cạnh qua từng tầng
python scripts/train_ata.py --arch negat --no-edge-update ... --ckpt-dir ck_negat_noeu
```

---

## 4. BƯỚC 3 — Đánh giá đa lớp VÀ nhị phân

```bash
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_gat_ata --binary --out kq_gat_ata.json
python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_negat_ata --binary --out kq_negat_ata.json

python scripts/test_all.py --data-dir data/rplguard_chrono --ckpt-dir checkpoints
```

Cờ `--binary` in thêm bảng gộp mọi lớp tấn công thành một lớp dương, kèm **tỉ lệ báo động sai** và **tỉ lệ bỏ lọt** — hai đại lượng mà macro-F1 đa lớp không thể hiện trực tiếp.

`train_ata.py` cũng tự in bảng nhị phân ở cuối mỗi lần chạy.

**Cách đọc đúng.** Chỉ số nhị phân **luôn** cao hơn macro-F1 đa lớp, vì phép gộp xoá bỏ toàn bộ lỗi nhầm **giữa** các loại tấn công: một mô hình nhầm `rank` thành `sinkhole` bị phạt ở chế độ đa lớp nhưng được tính đúng ở chế độ nhị phân. Phải báo cáo cả hai cạnh nhau, không dùng riêng con số nhị phân.

---

## 5. Trực quan hoá và chứng minh

### 5a. So sánh không gian đặc trưng CUỐI giữa các mô hình

```bash
python scripts/tsne_compare.py --data-dir data/radar_chrono \
  --models "GAT-IDS=ck_gat_noata" "GAT-IDS+ATA=ck_gat_ata" \
           "NE-GAT=ck_negat_noata" "NE-GAT+ATA=ck_negat_ata" \
  --n-samples 4000 --out tsne_4models.png --csv silhouette.csv
```

Mọi mô hình dùng **chung** một tập mẫu con phân tầng và **chung** hạt giống t-SNE, nên khác biệt nhìn thấy là do mô hình chứ không do phép lấy mẫu. Điểm silhouette **luôn** tính trên không gian gốc.

### 5b. Trực quan hoá theo TỪNG TẦNG

```bash
python scripts/tsne_layers.py --data-dir data/radar_chrono \
       --ckpt-dir ck_gat_ata --n-samples 4000 --out tsne_layers_gat.png
python scripts/tsne_layers.py --data-dir data/radar_chrono \
       --ckpt-dir ck_negat_ata --n-samples 4000 --out tsne_layers_negat.png
```

*Một khác biệt cần nêu rõ khi trình bày:* GAT-IDS phơi ra bốn vị trí trích (sau mỗi tầng chú ý và trước bộ phân loại), còn NE-GAT-IDS chỉ phơi ra biểu diễn **trước bộ phân loại**. Đây là hệ quả của kiến trúc hai kênh — biểu diễn tầng giữa là hợp nhất của hai kênh nên không có nghĩa "một tầng chú ý" như ở GAT-IDS. Không nên trình bày sự khác biệt này như một thiếu sót.

### 5c. Ma trận nhầm lẫn và khả diễn giải

```bash
python scripts/confusion_matrix.py --data-dir data/radar_chrono --ckpt-dir ck_gat_ata
python scripts/attention_per_class.py --data-dir data/radar_chrono \
       --ckpt-dir ck_gat_ata --top 10 --export-model-csv model_xai.csv
python scripts/feature_importance_per_class.py --data-dir data/radar_chrono \
       --top 10 --out dactrung_node.csv
python scripts/xai_simple.py --model-csv model_xai.csv \
       --rf-node-csv dactrung_node.csv --out-dir hinh_xai
```

---

## 6. Quét siêu tham số ATA

```bash
for K in 2 3 4 5; do
  for LD in 0.01 0.1 0.5 1.0; do
    for D in coral mmd cosine; do
      python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
        --K $K --lambda-d $LD --disc $D --epochs 120 \
        --ckpt-dir ck_K${K}_L${LD}_${D}
    done
  done
done
```

Thêm `--domain-strategy tdc` để so với chiến lược chia miền theo phân vị.

---

## 7. Nhiều hạt giống — việc bắt buộc trước khi nộp

```bash
for S in 42 43 44 45 46; do
  python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
    --K 3 --lambda-d 0.5 --disc mmd --seed $S --epochs 120 \
    --ckpt-dir ck_negat_s$S
  python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_negat_s$S --binary --out kq_negat_s$S.json
done
python scripts/aggregate_seeds.py --pattern "kq_negat_s*.json"
```

Với một hạt giống, không có cách nào biết một chênh lệch là thật hay chỉ là dao động ngẫu nhiên. Đây là lỗ hổng lớn nhất còn lại.

--- (Edge)GAT-IDS
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 42 --epochs 120 --ckpt-dir ck_gat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 43 --epochs 120 --ckpt-dir ck_gat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 44 --epochs 120 --ckpt-dir ck_gat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 45 --epochs 120 --ckpt-dir ck_gat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 46 --epochs 120 --ckpt-dir ck_gat_ata_radar
---

--- NEGAT-IDS
------------- RADAR DATASET
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 42 --epochs 120 --ckpt-dir ck_negat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 43 --epochs 120 --ckpt-dir ck_negat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 44 --epochs 120 --ckpt-dir ck_negat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 45 --epochs 120 --ckpt-dir ck_negat_ata_radar
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 46 --epochs 120 --ckpt-dir ck_negat_ata_radar
------------- UOS DATASET
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 42 --epochs 120 --ckpt-dir ck_negat_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 43 --epochs 120 --ckpt-dir ck_negat_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 44 --epochs 120 --ckpt-dir ck_negat_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 45 --epochs 120 --ckpt-dir ck_negat_ata_uos
python scripts/train_ata.py --data-dir data/uos_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 46 --epochs 120 --ckpt-dir ck_negat_ata_uos
------------- RPL-BEH DATASET
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 42 --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 43 --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 44 --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 45 --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
python scripts/train_ata.py --data-dir data/rplbeh_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 46 --epochs 120 --ckpt-dir ck_negat_ata_rplbeh
---
------------- RPLGUARD DATASET
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 42 --epochs 120 --ckpt-dir ck_negat_ata_rplguard_seed42
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 43 --epochs 120 --ckpt-dir ck_negat_ata_rplguard_seed43
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 44 --epochs 120 --ckpt-dir ck_negat_ata_rplguard_seed44
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 45 --epochs 120 --ckpt-dir ck_negat_ata_rplguard_seed45
python scripts/train_ata.py --data-dir data/rplguard_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --seed 46 --epochs 120 --ckpt-dir ck_negat_ata_rplguard_seed46
---
---


## 8. Trình tự tối thiểu cho một bộ dữ liệu

```bash
D=radar                      # đổi thành uos hoặc rplbeh
python scripts/convert_data.py --dataset $D --data-dir data/$D \
       --split chrono --alpha 1.0 --out-dir data/${D}_chrono
python scripts/train_all.py --data-dir data/${D}_chrono \
       --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_${D}_trees
for A in mlp gcn egraphsage edgegat negat; do
  python scripts/train_ata.py --data-dir data/${D}_chrono --arch $A \
    --K 3 --lambda-d 0.1 --epochs 120 --ckpt-dir ck_${D}_${A}
done
python scripts/test_all.py --data-dir data/${D}_chrono \
       --ckpt-dir ck_${D}_edgegat --binary
python scripts/tsne_compare.py --data-dir data/${D}_chrono \
  --models "GAT-IDS=ck_${D}_edgegat" "NE-GAT=ck_${D}_negat" \
  --out tsne_${D}.png
```




##################### GAT-TRANSFORMER-IDS 

Giảm tham số
Cấu hình	Tổng	Khối MAE	Giảm
--d-model 64 --dim-ff 128 --tr-layers 2	237.936	171.616	—
--d-model 32 --dim-ff 64 --tr-layers 2 (mặc định mới)	111.184	44.864	53%
--d-model 32 --dim-ff 64 --tr-layers 1	89.808	23.488	62%
--d-model 24 --dim-ff 48 --tr-layers 1	80.104	13.784	66%

#--> For RADAR DATASET
# Giai đoạn 1 — tự giám sát, chỉ đồ thị thuần bình thường
python scripts/train_ssl.py --data-dir data/radar_chrono --stage ssl --backbone negat --epochs 60 --ckpt-dir ck_ssl_radar --d-model 32 --dim-ff 64 --tr-layers 1

# Giai đoạn 2 — tinh chỉnh
python scripts/train_ssl.py --data-dir data/radar_chrono --stage finetune --backbone negat --init-from ck_ssl_radar --epochs 120 --ckpt-dir ck_gtids_radar --d-model 32 --dim-ff 64 --tr-layers 1

# Nhánh đối chứng — KHÔNG tiền huấn luyện
python scripts/train_ssl.py --data-dir data/radar_chrono --stage scratch --backbone negat --epochs 120 --ckpt-dir ck_gtids_scratch



#--> For RPLGUARD DATASET
# Giai đoạn 1 — tự giám sát, chỉ đồ thị thuần bình thường
python scripts/train_ssl.py --data-dir data/rplguard_chrono --stage ssl --backbone negat --epochs 60 --ckpt-dir ck_ssl_rplguard --d-model 32 --dim-ff 64 --tr-layers 1

# Giai đoạn 2 — tinh chỉnh
python scripts/train_ssl.py --data-dir data/rplguard_chrono --stage finetune --backbone negat --init-from ck_ssl_rplguard --epochs 120 --ckpt-dir ck_gtids_rplguard --d-model 32 --dim-ff 64 --tr-layers 1

# Nhánh đối chứng — KHÔNG tiền huấn luyện
python scripts/train_ssl.py --data-dir data/rplguard_chrono --stage scratch --backbone negat --epochs 120 --ckpt-dir ck_gtids_scratch_rplguard --d-model 32 --dim-ff 64 --tr-layers 1




## Tham khảo chính

- Winter và cộng sự (2012). *RPL: IPv6 Routing Protocol for Low-Power and Lossy Networks.* RFC 6550.
- Brody, Alon, Yahav (2022). *How Attentive are Graph Attention Networks?* ICLR.
- Veličković và cộng sự (2018). *Graph Attention Networks.* ICLR.
- Kipf, Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Lo và cộng sự (2022). *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT.* IEEE/IFIP NOMS.
- Cui và cộng sự (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
- Lin và cộng sự (2017). *Focal Loss for Dense Object Detection.* ICCV.
- Xu, Hu, Leskovec, Jegelka (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Corso và cộng sự (2020). *Principal Neighbourhood Aggregation for Graph Nets.* NeurIPS.
- Izmailov và cộng sự (2018). *Averaging Weights Leads to Wider Optima and Better Generalization.* UAI.
- Yeo, Johnson (2000). *A New Family of Power Transformations to Improve Normality or Symmetry.* Biometrika.
- Rousseeuw (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* J. Comput. Appl. Math.
- Phan, Dang Le, Vu, Vo (2026). *Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift.*

---

## Tác giả

**Đồng Duy Khang** — MSHV 2470165, Kỹ thuật Viễn thông
Hướng dẫn: **TS. Võ Quế Sơn**
Khoa Điện – Điện tử, Trường Đại học Bách khoa, ĐHQG TP.HCM

---

## Giấy phép

*(Chưa chọn — cân nhắc MIT hoặc Apache-2.0. Lưu ý các bộ dữ liệu có giấy phép riêng của tác giả gốc và không được phân phối lại trong kho này.)*
