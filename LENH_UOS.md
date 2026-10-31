# Lệnh chạy UOS — đã sửa lỗi

## Lỗi trong lệnh tôi đưa trước đó

Lệnh cũ:
```
--split-by-file "2-Single_Attacker:1-Normal_Traffic:3-Dual_Attackers" --split-key scenario
```

**Không lỗi cú pháp** — nó chạy được. Nhưng có LỖI LOGIC nghiêm trọng hơn:

```
train  344 đồ thị | {Normal: 4350, Sinkhole: 2113}
val     29 đồ thị | {Normal: 566}          ← CHỈ MỘT LỚP
test    34 đồ thị | {Normal: 266, Sinkhole: 282}
```

Nhóm `1-Normal_Traffic` chỉ chứa lưu lượng bình thường, không có node tấn công
nào. Đặt nó làm tập kiểm định khiến macro-F1 kiểm định **chỉ tính trên một lớp**
→ cơ chế chọn điểm kiểm tra hỏng hoàn toàn: mô hình "tốt nhất" được chọn theo
một chỉ số không phân biệt được gì.

Nếu đảo `1-Normal_Traffic` sang tập kiểm thử thì lỗi chuyển sang tập đó.

**Nguyên nhân gốc:** nhóm `1-Normal_Traffic` KHÔNG BAO GIỜ được đứng một mình
làm tập kiểm định hay kiểm thử.

## Đã sửa trong code

1. **Kiểm tra tự động**: `split_uos_by_scenario()` giờ báo lỗi rõ ràng nếu tập
   nào chỉ có một lớp.
2. **Cho phép ghép nhóm bằng dấu `+`**: `"2-Single_Attacker:3-Dual_Attackers+1-Normal_Traffic:..."`

---

## LỆNH ĐÚNG — chia theo TỆP (khuyến nghị)

Mỗi tập đều có cả tệp Normal lẫn tệp tấn công, và tập kiểm thử chứa kẻ tấn
công ở vị trí CHƯA TỪNG THẤY khi huấn luyện.

### Linux / macOS / Git Bash

```bash
TR="SingleAttacker_Small_SingleDODAG_2,SingleAttacker_Small_SingleDODAG_3,SingleAttacker_Small_SingleDODAG_4,SingleAttacker_Small_SingleDODAG_5,SingleAttacker_Small_SingleDODAG_6,SingleAttacker_Small_SingleDODAG_7,SingleAttacker_Medium_DualDODAG_7,SingleAttacker_Medium_DualDODAG_8,SingleAttacker_Medium_DualDODAG_9,Normal_Small_SingleDODAG"

VA="SingleAttacker_Small_SingleDODAG_8,SingleAttacker_Medium_DualDODAG_10,SingleAttacker_Medium_DualDODAG_12"

TE="SingleAttacker_Small_SingleDODAG_9,SingleAttacker_Small_SingleDODAG_10,SingleAttacker_Small_SingleDODAG_11,SingleAttacker_Small_SingleDODAG_12,SingleAttacker_Small_SingleDODAG_13,SingleAttacker_Medium_DualDODAG_13,SingleAttacker_Medium_DualDODAG_15,SingleAttacker_Medium_DualDODAG_16,SingleAttacker_Medium_DualDODAG_17,SingleAttacker_Medium_DualDODAG_19,SingleAttacker_Medium_DualDODAG_20,DualAttackers_Small_SingleDODAG_11_12,DualAttackers_Small_SingleDODAG_5_12,DualAttackers_Small_SingleDODAG_5_9,DualAttackers_Small_SingleDODAG_8_2,Normal_Medium_DualDODAG"

python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
    --split-by-file "$TR:$VA:$TE" --split-key src_file
```

### Windows CMD — biến dùng %VAR%, KHÔNG dùng $VAR

```cmd
set TR=SingleAttacker_Small_SingleDODAG_2,SingleAttacker_Small_SingleDODAG_3,SingleAttacker_Small_SingleDODAG_4,SingleAttacker_Small_SingleDODAG_5,SingleAttacker_Small_SingleDODAG_6,SingleAttacker_Small_SingleDODAG_7,SingleAttacker_Medium_DualDODAG_7,SingleAttacker_Medium_DualDODAG_8,SingleAttacker_Medium_DualDODAG_9,Normal_Small_SingleDODAG

set VA=SingleAttacker_Small_SingleDODAG_8,SingleAttacker_Medium_DualDODAG_10,SingleAttacker_Medium_DualDODAG_12

set TE=SingleAttacker_Small_SingleDODAG_9,SingleAttacker_Small_SingleDODAG_10,SingleAttacker_Small_SingleDODAG_11,SingleAttacker_Small_SingleDODAG_12,SingleAttacker_Small_SingleDODAG_13,SingleAttacker_Medium_DualDODAG_13,SingleAttacker_Medium_DualDODAG_15,SingleAttacker_Medium_DualDODAG_16,SingleAttacker_Medium_DualDODAG_17,SingleAttacker_Medium_DualDODAG_19,SingleAttacker_Medium_DualDODAG_20,DualAttackers_Small_SingleDODAG_11_12,DualAttackers_Small_SingleDODAG_5_12,DualAttackers_Small_SingleDODAG_5_9,DualAttackers_Small_SingleDODAG_8_2,Normal_Medium_DualDODAG

python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 --split-by-file "%TR%:%VA%:%TE%" --split-key src_file
```

### Hoặc viết thẳng một dòng (an toàn nhất, không dùng biến)

```
python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 --split-key src_file --split-by-file "SingleAttacker_Small_SingleDODAG_2,SingleAttacker_Small_SingleDODAG_3,SingleAttacker_Small_SingleDODAG_4,SingleAttacker_Small_SingleDODAG_5,SingleAttacker_Small_SingleDODAG_6,SingleAttacker_Small_SingleDODAG_7,SingleAttacker_Medium_DualDODAG_7,SingleAttacker_Medium_DualDODAG_8,SingleAttacker_Medium_DualDODAG_9,Normal_Small_SingleDODAG:SingleAttacker_Small_SingleDODAG_8,SingleAttacker_Medium_DualDODAG_10,SingleAttacker_Medium_DualDODAG_12:SingleAttacker_Small_SingleDODAG_9,SingleAttacker_Small_SingleDODAG_10,SingleAttacker_Small_SingleDODAG_11,SingleAttacker_Small_SingleDODAG_12,SingleAttacker_Small_SingleDODAG_13,SingleAttacker_Medium_DualDODAG_13,SingleAttacker_Medium_DualDODAG_15,SingleAttacker_Medium_DualDODAG_16,SingleAttacker_Medium_DualDODAG_17,SingleAttacker_Medium_DualDODAG_19,SingleAttacker_Medium_DualDODAG_20,DualAttackers_Small_SingleDODAG_11_12,DualAttackers_Small_SingleDODAG_5_12,DualAttackers_Small_SingleDODAG_5_9,DualAttackers_Small_SingleDODAG_8_2,Normal_Medium_DualDODAG"
```

### Kết quả kiểm chứng

```
train  128 đồ thị | {Normal: 1703, Sinkhole:  585}
val     52 đồ thị | {Normal:  605, Sinkhole:  370}
test   227 đồ thị | {Normal: 2874, Sinkhole: 1440}
Không phát hiện rò rỉ theo cả hai tiêu chí.
```

Cả ba tập đều đủ hai lớp.

---

## Nếu vẫn muốn chia theo NHÓM KỊCH BẢN

Phải GHÉP `1-Normal_Traffic` vào một nhóm khác bằng dấu `+`:

```bash
python scripts/convert_data.py --dataset uos --data-dir data/uos --window 10 \
    --split-key scenario \
    --split-by-file "2-Single_Attacker:3-Dual_Attackers:1-Normal_Traffic+3-Dual_Attackers"
```

Nhưng cách này KHÔNG chạy được vì `3-Dual_Attackers` xuất hiện ở hai tập
(hàm sẽ báo lỗi trùng). Với bộ UOS chỉ có 3 nhóm và một nhóm thuần Normal,
**chia theo TỆP là phương án khả thi duy nhất**. Đó cũng là phương án nghiêm
ngặt hơn, vì nó kiểm tra khái quát hoá sang VỊ TRÍ KẺ TẤN CÔNG mới.

---

## Huấn luyện và kiểm thử

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3
python scripts/test_all.py
```

Lưu ý: UOS chỉ có 2 lớp nên macro-F1 ở đây không so sánh trực tiếp được với
macro-F1 16 lớp của RADAR. Trong luận văn nên trình bày riêng, kèm ghi chú.
