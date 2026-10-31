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

---

## Bộ dữ liệu

Kho này **không phân phối lại dữ liệu**. Ba bộ đều là dữ liệu bên thứ ba, cần tải từ nguồn gốc.

| Bộ | Số lớp | Ghi chú |
|:---|---:|:---|
| **RADAR** | 16 | Bộ chính. Duy nhất có đủ bốn nhóm thông tin: độ lệch rank, độ lệch phiên bản, thống kê chuyển tiếp, chất lượng liên kết |

| **RPL-Beh** | 

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
- Phan, Dang Le, Vu, Vo (2026). *Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift.*

---

## Tác giả

**Đồng Duy Khang** — Kỹ thuật Viễn thông
Hướng dẫn: **TS. Võ Quế Sơn**
Khoa Điện – Điện tử, Trường Đại học Bách khoa, ĐHQG TP.HCM

---

## Giấy phép

*(Chưa chọn — cân nhắc MIT hoặc Apache-2.0. Lưu ý các bộ dữ liệu có giấy phép riêng của tác giả gốc và không được phân phối lại trong kho này.)*
