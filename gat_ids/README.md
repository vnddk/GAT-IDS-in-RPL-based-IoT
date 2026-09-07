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



# GAT-IDS · NE-GAT-IDS

**Phát hiện xâm nhập trong mạng IoT định tuyến RPL bằng mạng nơ-ron đồ thị có chú ý, huấn luyện với căn chỉnh miền thời gian thích nghi (ATA).**

Khung phần mềm cho luận văn thạc sĩ — Khoa Điện – Điện tử, Trường Đại học Bách khoa, ĐHQG TP.HCM.

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

Sau Bước 1, **kiểm ngay hai dòng trong log**:

- `Chẩn đoán rò rỉ thời gian: δ̄ = …` — phải lớn hơn 2 rõ rệt.
- `Không phát hiện rò rỉ` hoặc `WARNING — Đặc trưng RÒ RỈ` — nếu có cảnh báo, phải xử lý trước khi huấn luyện.

Xem [`LENH_CHAY_ATA.md`](LENH_CHAY_ATA.md) cho toàn bộ luồng lệnh, gồm quét siêu tham số, ablation và chạy nhiều hạt giống.

---

## Cấu trúc mã nguồn

```
src/
├── data/
│   ├── radar_builder.py       Dựng đồ thị DODAG từ nhật ký gói tin RADAR
│   ├── uos_builder.py         Bộ UOS — xử lý ba schema CSV khác nhau
│   ├── rplbeh_builder.py      Bộ RPL-Beh — đảo hoán vị seed 42, cắt lại 95 mô phỏng
│   ├── iotrpl_builder.py      Bộ IoT-RPL (đã loại khỏi kết luận, xem bên dưới)
│   ├── feature_audit.py       Kiểm toán rò rỉ nhãn hai tiêu chí
│   ├── transforms.py          Yeo–Johnson, chỉ fit trên tập huấn luyện
│   └── loader.py              Đăng ký bộ dữ liệu, chia tập, tái lấy mẫu
├── models/
│   ├── edge_gat.py            GAT-IDS
│   ├── ne_gat.py              NE-GAT-IDS hai kênh
│   ├── gnn_baselines.py       MLP, GCN, E-GraphSAGE
│   └── tabular_baselines.py   Random Forest, XGBoost, LightGBM
└── utils/
    ├── ata.py                 Độ đo chênh lệch, dựng miền, chia theo thời gian, δ̄
    ├── binary_eval.py         Đánh giá nhị phân
    ├── losses.py              Class-Balanced Focal có chặn trần trọng số
    ├── engine.py              Vòng lặp huấn luyện, EMA, dừng sớm
    └── metrics.py             macro-F1, báo cáo theo lớp

scripts/
├── convert_data.py                  Bước 1
├── train_all.py                     Bước 2 — bảy mô hình, không ATA
├── train_ata.py                     Bước 2b — huấn luyện có ATA
├── test_all.py                      Bước 3 — đánh giá đa lớp + nhị phân
├── tsne_compare.py                  So sánh không gian đặc trưng cuối
├── tsne_layers.py                   Trực quan hoá theo từng tầng
├── attention_per_class.py           Trọng số chú ý + GNNExplainer
├── feature_importance_per_class.py  Tầm quan trọng đặc trưng theo lớp
├── xai_simple.py                    Biểu đồ đối chiếu mô hình với dữ liệu
├── ablation_edge_attention.py       Ablation đặc trưng cạnh
└── PATCH_bo_ro_ri.py                Loại đặc trưng bị kiểm toán gắn cờ
```

---

## Bộ dữ liệu

Kho này **không phân phối lại dữ liệu**. Ba bộ đều là dữ liệu bên thứ ba, cần tải từ nguồn gốc.

| Bộ | Số lớp | Ghi chú |
|:---|---:|:---|
| **RADAR** | 16 | Bộ chính. Duy nhất có đủ bốn nhóm thông tin: độ lệch rank, độ lệch phiên bản, thống kê chuyển tiếp, chất lượng liên kết |
| **UOS_IOTSH_2024** | 2 | 49 trên 78 tệp công bố chỉ chứa **hai byte** và không có dữ liệu; builder tự bỏ qua |
| **RPL-Beh** | 5 | CSV đã bị xáo trộn bằng `df.sample(random_state=42)`; builder đảo ngược được và cắt lại 95 mô phỏng |
| **IoT-RPL 2021** | 5 | **Đã loại khỏi mọi kết luận** — xem bên dưới |

### Vì sao IoT-RPL bị loại

Kiểm toán gắn cờ đặc trưng `send_share` với F1 gốc-cây một cột đạt 0,914. Sau khi loại đặc trưng đó và huấn luyện lại, **ba họ mô hình khác hẳn nhau về nguyên lý** vẫn cùng đạt độ chính xác tuyệt đối 1,0000.

Nguyên nhân gốc nằm ở **thiết kế** bộ dữ liệu: mỗi nút tấn công giữ nguyên một vai trò ở mọi tệp mô phỏng, nên bất kỳ tổ hợp đặc trưng nào đủ nhận ra danh tính một nút cũng tiết lộ nhãn. Có một lập luận khép kín: cách xử lý chuẩn mực là chia theo **thực thể**, nhưng mỗi loại tấn công chỉ gắn với đúng một nút, nên giữ nút đó ra khỏi tập huấn luyện đồng nghĩa lớp đó biến mất. Phép chia đúng đắn **bất khả thi về mặt cấu trúc**, và chính sự bất khả thi ấy là bằng chứng.

### Về việc khôi phục thứ tự thời gian của RPL-Beh

`dataset_generation.py` của tác giả gốc kết thúc bằng `df.sample(frac=1, random_state=42)`. pandas dùng `numpy.RandomState` nên hoán vị tái lập được chính xác. Sau khi đảo, `node_id` tăng nghiêm ngặt trong mọi interval:

| | Nguyên trạng | Sau khi đảo |
|:---|---:|---:|
| Tỉ lệ `node_id` tăng khi cùng `time_sec` | 0,4951 | **1,000000** |
| Số vi phạm đơn điệu | — | **0 / 151.224** |

Ngoài ra, cả 95/95 mô phỏng có cửa sổ cách đều đúng 8 đơn vị, từ 8 tới 592. `time_sec` là chỉ số **phút**, nên 8 phút = 480 giây = đúng Capture Interval trong bài báo gốc. Đây là một miền thời gian liên tục, dùng được cho ATA.

**Hạn chế còn lại của bộ này:** đặc trưng láng giềng đã bị tác giả gốc lấy trung bình (`np.mean` trong `generate_csv()`), nên thông tin theo **từng** láng giềng không khôi phục được từ CSV. Vì vậy bộ này không phù hợp để chứng minh giá trị của kênh cạnh.

---

## Kết quả

Bộ RADAR, chia theo thời gian, δ̄ = 60,5. Tập kiểm thử 5.220 đồ thị / 98.579 nút. Mọi mô hình dùng cùng tập, cùng phép chia, cùng hạt giống.

| # | Mô hình | Macro-F1 | Accuracy | Tham số |
|:--|:---|---:|---:|---:|
| 1 | **NE-GAT-IDS + ATA** | **0,9440** | 0,9773 | 62.224 |
| 2 | GAT-IDS + ATA | 0,9358 | 0,9724 | 103.956 |
| 3 | E-GraphSAGE + ATA | 0,8978 | 0,9527 | 10.834 |
| 4 | XGBoost | 0,8892 | 0,9674 | cây |
| 5 | Random Forest | 0,8826 | 0,9593 | cây |
| 6 | MLP + ATA | 0,7689 | 0,9117 | 6.147 |
| 7 | GCN + ATA | 0,7454 | 0,8733 | 3.698 |
| 8 | LightGBM | 0,4728 | 0,8221 | cây |

Kết quả tách thành **ba nhóm rõ rệt**: đồ thị có đặc trưng cạnh (0,90–0,94), mô hình cây (0,88–0,89), nơ-ron không dùng cạnh (0,75–0,77).

Điểm đáng chú ý nhất: **GCN có cấu trúc đồ thị nhưng thua cả MLP**. Không phải cứ có cấu trúc đồ thị là tốt — điều tạo khác biệt là cơ chế chú ý học được cộng với đặc trưng liên kết.

Hai lưu ý về cách đọc bảng. XGBoost có accuracy **cao hơn** E-GraphSAGE nhưng macro-F1 **thấp hơn**: accuracy thưởng cho việc đoán tốt lớp đa số, macro-F1 phạt việc bỏ sót lớp thiểu số. Còn LightGBM 0,4728 là hệ quả của việc tắt tái lấy mẫu để đồng nhất giao thức — nó là mô hình duy nhất không còn cơ chế xử lý mất cân bằng nào, chứ không phải giới hạn của thuật toán.

---

## Giới hạn hiện tại

Phần này liệt kê những gì khung này **chưa** chứng minh được. Đọc trước khi trích dẫn bất kỳ con số nào ở trên.

**Một hạt giống duy nhất.** Mọi con số đến từ một lần chạy. Chênh lệch giữa NE-GAT-IDS và GAT-IDS là 0,0082 điểm macro-F1 — chưa phân biệt được hiệu ứng thật với dao động ngẫu nhiên. Cần tối thiểu năm hạt giống và báo cáo độ lệch chuẩn.

**Chưa có nhánh đối chứng không ATA.** Chưa chạy `--K 1 --lambda-d 0` trên cùng tập chia theo thời gian, nên **chưa đo được đóng góp riêng của ATA** — chỉ biết cả hai mô hình đều hoạt động tốt khi *có* ATA.

**Chênh lệch dung lượng 41%.** NE-GAT-IDS thắng với ít tham số hơn nên kết luận mạnh hơn. Nhưng để loại hẳn yếu tố này, cần thêm biến thể nâng chiều cho tương đương tham số.

**Chỉ có bộ RADAR.** UOS và RPL-Beh đã chuyển đổi xong nhưng chưa huấn luyện đầy đủ.

**Chỉ dữ liệu mô phỏng.** Chưa có bằng chứng trên phần cứng thực với nhiễu vô tuyến và mất gói thật.

**Biểu diễn theo cửa sổ làm mất thứ tự sự kiện.** Phép gộp giữ các đại lượng bất biến thứ tự nhưng mất trình tự trong cửa sổ, nên các tấn công có chữ ký thuần tuý phụ thuộc trình tự ở thế bất lợi. Kết quả nhất quán với nhận định này: `delayed_reply` có F1 thấp nhất ở cả hai mô hình.

**Nguồn công bố của RADAR và UOS chưa xác minh.** Hai bộ này được dùng làm dữ liệu chính nhưng tài liệu gốc chưa được xác minh; không nên trích dẫn phỏng đoán.

**EMA có thiên lệch khởi tạo.** Bộ đệm EMA khởi tạo từ trọng số ngẫu nhiên với ρ = 0,995. Với tập nhỏ như UOS (4 bước/epoch), sau 20 epoch vẫn còn 67% trọng số khởi tạo, gây hiện tượng chỉ số kiểm định phẳng giả tạo ở các epoch đầu. Khắc phục bằng hiệu chỉnh thiên lệch kiểu Adam, nhưng việc đó sẽ thay đổi mọi con số đã báo cáo.

---

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

Hướng dẫn: **TS. Võ Quế Sơn và cộng sự**
Khoa Điện – Điện tử, Trường Đại học Bách khoa, ĐHQG TP.HCM

---

## Giấy phép

*(Chưa chọn — cân nhắc MIT hoặc Apache-2.0. Lưu ý các bộ dữ liệu có giấy phép riêng của tác giả gốc và không được phân phối lại trong kho này.)*

python scripts/test_all.py --data-dir data/${D}_chrono \
       --ckpt-dir ck_${D}_edgegat --binary
python scripts/tsne_compare.py --data-dir data/${D}_chrono \
  --models "GAT-IDS=ck_${D}_edgegat" "NE-GAT=ck_${D}_negat" \
  --out tsne_${D}.png
```
