# So sánh ba bộ dữ liệu: RADAR · IoT-RPL 2021 · UOS_IOTSH_2024

## Tóm tắt

| | RADAR | IoT-RPL 2021 | **UOS_IOTSH_2024** |
|---|---|---|---|
| Số lớp | 16 | 5 | 2 (Normal/Sinkhole) |
| Cột thời gian | có | **KHÔNG** | **có** |
| RPL Rank | có | **KHÔNG** | **có** |
| Đích gói | 1 node | danh sách | 1 node |
| Số node | ~18 | 16 cố định | 12 hoặc 24 |
| Vị trí kẻ tấn công | cố định/tệp | **CỐ ĐỊNH mọi tệp** | **THAY ĐỔI theo tệp** |
| Kiểm toán rò rỉ | sạch | ⚠ F1 một cột 0,906 | **sạch (0,462)** |

**UOS là bộ dữ liệu thứ hai TỐT NHẤT** trong ba bộ, vì nó khắc phục đúng điểm
yếu chí tử của IoT-RPL.

---

## Vì sao UOS giá trị hơn IoT-RPL

### IoT-RPL: node tấn công CỐ ĐỊNH
Node 12 là Blackhole ở **mọi** tệp, node 14 là Rank ở mọi tệp. Mô hình có thể
chỉ học "node 12 = Blackhole". Kiểm toán gắn cờ `send_share` F1 = 0,906.

### UOS: vị trí kẻ tấn công THAY ĐỔI
Tên tệp mã hoá ID node tấn công:

```
SingleAttacker_Small_SingleDODAG_2.csv    -> node 2 tấn công
SingleAttacker_Small_SingleDODAG_13.csv   -> node 13 tấn công
DualAttackers_Small_SingleDODAG_11_12.csv -> node 11 và 12 tấn công
```

Bộ dữ liệu chạy **vét cạn** mọi node làm kẻ tấn công. Nhờ đó có thể chia tập
sao cho tập kiểm thử chứa kẻ tấn công ở **vị trí chưa từng thấy khi huấn luyện**
— đúng yêu cầu triển khai thực tế.

Kiểm toán trên phép chia này: F1 một cột cao nhất **0,462**, không đặc trưng
nào bị gắn cờ. Đây là bằng chứng cho thấy mô hình buộc phải học HÀNH VI, không
thể học danh tính node.

---

## ⚠ Khuyết tật của bộ UOS gốc: 49/78 tệp RỖNG

Kiểm tra kích thước: **49 tệp chỉ có đúng 2 byte** (một ký tự xuống dòng).

| Nhóm | Tệp có dữ liệu |
|---|---|
| 1-Normal_Traffic | 2/3 |
| 2-Single_Attacker | 23/60 |
| 3-Dual_Attackers | 4/15 |

Toàn bộ nhóm `Medium_SingleDODAG` (24 node, một DODAG) rỗng hoàn toàn. Đây là
khuyết tật của bộ dữ liệu công bố, không phải lỗi đọc tệp. Còn lại **29 tệp**
dùng được, cho **407 đồ thị**.

Cần nêu rõ trong luận văn — và nên báo cho nhóm tác giả bộ dữ liệu.

---

## Đã sửa trong builder

`uos_builder.py` có sẵn nhưng dùng `glob("*.csv")` quét thư mục PHẲNG, trong
khi UOS có cấu trúc BA TẦNG → trả về **0 đồ thị**. Đã sửa thành quét đệ quy,
đồng thời gắn `src_file` và `scenario` cho mỗi đồ thị.

Kết quả sau khi sửa: 29 tệp OK, 407 đồ thị, d_n=13, d_e=10, không NaN/Inf,
phân bố node Normal 5.182 / Sinkhole 2.395.

---

## Lệnh chạy

### Chia theo TỆP — nghiêm ngặt nhất (khuyến nghị)

Tập kiểm thử chứa kẻ tấn công ở vị trí chưa từng thấy khi huấn luyện:

```bash
TR="SingleAttacker_Small_SingleDODAG_2,SingleAttacker_Small_SingleDODAG_3,SingleAttacker_Small_SingleDODAG_4,SingleAttacker_Small_SingleDODAG_5,SingleAttacker_Small_SingleDODAG_6,SingleAttacker_Small_SingleDODAG_7,SingleAttacker_Medium_DualDODAG_7,SingleAttacker_Medium_DualDODAG_8,SingleAttacker_Medium_DualDODAG_9,SingleAttacker_Medium_DualDODAG_10,Normal_Small_SingleDODAG,Normal_Medium_DualDODAG"
VA="SingleAttacker_Small_SingleDODAG_8,SingleAttacker_Medium_DualDODAG_12,DualAttackers_Small_SingleDODAG_5_9"
TE="SingleAttacker_Small_SingleDODAG_9,SingleAttacker_Small_SingleDODAG_10,SingleAttacker_Small_SingleDODAG_11,SingleAttacker_Small_SingleDODAG_12,SingleAttacker_Small_SingleDODAG_13,SingleAttacker_Medium_DualDODAG_13,SingleAttacker_Medium_DualDODAG_15,SingleAttacker_Medium_DualDODAG_16,SingleAttacker_Medium_DualDODAG_17,SingleAttacker_Medium_DualDODAG_19,SingleAttacker_Medium_DualDODAG_20,DualAttackers_Small_SingleDODAG_11_12,DualAttackers_Small_SingleDODAG_5_12,DualAttackers_Small_SingleDODAG_8_2"

python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
    --split-by-file "$TR:$VA:$TE" --split-key src_file

python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3
python scripts/test_all.py
```

Kết quả chia: train 169 / val 39 / test 199 đồ thị.

### Chia theo NHÓM KỊCH BẢN (kiểm tra khái quát hoá mạnh hơn nữa)

```bash
python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
    --split-by-file "2-Single_Attacker:1-Normal_Traffic:3-Dual_Attackers" \
    --split-key scenario
```

Huấn luyện trên kịch bản MỘT kẻ tấn công, kiểm thử trên HAI kẻ tấn công —
đây là phép thử khái quát hoá rất khắt khe và là điểm cộng lớn cho luận văn.

---

## Đề xuất vị trí trong luận văn

| Bộ | Vai trò | Lý do |
|---|---|---|
| **RADAR** | Bộ CHÍNH | 16 lớp, đầy đủ nhất, có rank và next-hop |
| **UOS_IOTSH** | Bộ kiểm chứng KHÁI QUÁT HOÁ | vị trí kẻ tấn công thay đổi, kiểm toán sạch |
| IoT-RPL 2021 | Bộ kiểm chứng KHẢ CHUYỂN | cấu trúc rất khác (không thời gian, không rank) — nêu kèm hạn chế node cố định |

Ba bộ với ba vai trò khác nhau là cấu trúc thực nghiệm mạnh cho Chương 4:
một bộ đo hiệu năng, một bộ đo khái quát hoá, một bộ đo tính khả chuyển.
