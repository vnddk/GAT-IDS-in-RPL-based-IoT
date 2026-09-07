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
python scripts/train_all.py --data-dir data/uos_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_uos
python scripts/train_all.py --data-dir data/rplbeh_chrono --models RF,XGBoost,LightGBM --resample off --ckpt-dir ck_trees_rplbeh_chrono
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

### 2c. Hai mô hình chính

```bash
# GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_gat_ata_radar

# NE-GAT-IDS + ATA
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat --K 3 --lambda-d 0.5 --disc mmd --domain-strategy quantile --epochs 120 --ckpt-dir ck_negat_ata_radar
```

### 2d. Nhánh đối chứng KHÔNG có ATA

```bash
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat \
       --K 1 --lambda-d 0 --epochs 120 --ckpt-dir ck_gat_noata
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --K 1 --lambda-d 0 --epochs 120 --ckpt-dir ck_negat_noata
```

Đặt `--K 1 --lambda-d 0` là **có chủ ý**: cấu hình này thoái hoá về cực tiểu hoá rủi ro thực nghiệm thông thường nhưng **vẫn đi qua đúng một đường mã**, nên loại được mọi khác biệt cài đặt khỏi phép so sánh.

---

## 3. Ablation đặc trưng cạnh — hai câu hỏi khác nhau

Anh hỏi hai điều khác nhau, và chúng cần hai cặp lệnh khác nhau.

### 3a. "Có đặc trưng cạnh" so với "không có đặc trưng cạnh"

Áp cho **GAT-IDS**. Khác biệt duy nhất là cờ `--no-edge`:

```bash
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat \
       --K 3 --lambda-d 0.1 --epochs 120 --ckpt-dir ck_gat_edge
python scripts/train_ata.py --data-dir data/radar_chrono --arch edgegat \
       --K 3 --lambda-d 0.1 --epochs 120 --no-edge --ckpt-dir ck_gat_noedge
```

*Lưu ý:* cờ `--no-edge` hiện chỉ có trong `train_all.py`. Nếu `train_ata.py` báo lỗi không nhận cờ, hãy đặt `use_edge_features: false` trong `configs/default.yaml` trước khi chạy nhánh thứ hai, rồi đặt lại `true`.

### 3b. "Cạnh CÓ trong attention" so với "cạnh KHÔNG trong attention"

Đây là câu hỏi khác và chỉ áp được cho **NE-GAT-IDS**, vì kiến trúc này tách riêng kênh cạnh:

```bash
# cạnh tham gia điểm chú ý cùng ngữ cảnh hai đầu mút (mặc định)
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --edge-score full --K 3 --lambda-d 0.5 --disc mmd \
       --epochs 120 --ckpt-dir ck_negat_full

# điểm chú ý CHỈ tính từ trạng thái cạnh, cô lập khỏi biểu diễn nút
python scripts/train_ata.py --data-dir data/radar_chrono --arch negat \
       --edge-score edge_only --K 3 --lambda-d 0.5 --disc mmd \
       --epochs 120 --ckpt-dir ck_negat_edgeonly
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
python scripts/test_all.py --data-dir data/radar_chrono \
       --ckpt-dir ck_gat_ata --binary --out kq_gat_ata.json
python scripts/test_all.py --data-dir data/radar_chrono \
       --ckpt-dir ck_negat_ata --binary --out kq_negat_ata.json
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
  python scripts/test_all.py --data-dir data/radar_chrono \
    --ckpt-dir ck_negat_s$S --binary --out kq_negat_s$S.json
done
python scripts/aggregate_seeds.py --pattern "kq_negat_s*.json"
```

Với một hạt giống, không có cách nào biết một chênh lệch là thật hay chỉ là dao động ngẫu nhiên. Đây là lỗ hổng lớn nhất còn lại.

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
