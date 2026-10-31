# Ablation: tắt phần dư chính

## Đã hiện thực

Cờ `use_main_residual` (mặc định `true`, tương thích ngược hoàn toàn).

    Bật:  h_L ← ELU( h_L + W_res·h⁰ )     W_res: Linear(48→32)
    Tắt:  h_L ← ELU( h_L )                 giữ ELU để so sánh công bằng

Khi tắt, `res_proj` KHÔNG được khởi tạo nên checkpoint không chứa tham số chết.

| Cấu hình | Tham số | Chênh lệch |
|---|---|---|
| L=3, CÓ phần dư chính | 92.996 | — |
| L=3, TẮT phần dư chính | 91.428 | −1.568 = đúng Linear(48→32) |

## Hai cơ chế ĐỘC LẬP — đừng nhầm

| | Phần dư chính | Kết nối tắt đặc trưng gốc |
|---|---|---|
| Cờ | `use_main_residual` | `use_input_skip` |
| Nguồn | h⁰ — ĐÃ qua Linear+DyT+ELU | X — đặc trưng THÔ 29 chiều |
| Phép toán | **CỘNG** (giữ nguyên 32 chiều) | **NỐI** (32+29 = 61 chiều) |
| Vị trí | sau tầng GAT cuối | ngay trước bộ phân loại |
| Tham số | Linear(48→32) = 1.568 | 0 (chỉ nối) |

Tắt phần dư chính KHÔNG làm mất kết nối tắt. Sau khi tắt, mô hình vẫn còn
HAI đường bảo vệ thông tin riêng của node: phần dư từng tầng (nếu bật) và
kết nối tắt đặc trưng gốc.

## Lệnh chạy

```bash
# A. Mốc: cấu hình tốt nhất hiện tại (macro-F1 0,8621)
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3 --ckpt-dir ck_L3

# B. Y HỆT A nhưng TẮT phần dư chính  ← thí nghiệm của anh
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --resample off --weight-cap 3 --no-main-residual \
    --ckpt-dir ck_L3_nomainres

python scripts/test_all.py --ckpt-dir ck_L3_nomainres
python scripts/tsne_layers.py --ckpt-dir ck_L3_nomainres --n-samples 4000
```

Chỉ đổi ĐÚNG MỘT biến so với A, nên chênh lệch đo được là đóng góp thật của
phần dư chính.

## Cách đọc kết quả

Bốn lớp tấn công ĐƠN-NODE là nơi phần dư có tác dụng rõ nhất, vì chúng phụ
thuộc vào việc giữ được đặc trưng riêng của node khỏi bị trung bình hoá:

| Lớp | L=3 có phần dư |
|---|---|
| blackhole | 1,000 |
| selective_forward | 1,000 |
| worst_parent | 1,000 |
| local_repair | 1,000 |

Nếu sau khi tắt mà bốn lớp này TỤT, đó là bằng chứng trực tiếp cho vai trò
chống làm mịn quá mức của phần dư chính. Nếu chúng GIỮ NGUYÊN, kết luận là
kết nối tắt đặc trưng gốc và phần dư từng tầng đã đủ, và phần dư chính là
thành phần dư thừa — cũng là một kết quả đáng báo cáo.

Kèm theo, so điểm silhouette tầng cuối giữa hai cấu hình để có số đo khách
quan ngoài macro-F1.

## Bộ ablation đầy đủ cho luận văn

| Mã | num_layers | layer_res | main_res | input_skip |
|---|---|---|---|---|
| A | 3 | có | có | có |
| B | 3 | có | **KHÔNG** | có |
| C | 3 | **KHÔNG** | có | có |
| D | 3 | có | có | **KHÔNG** |
| E | 2 | — | có | có |

A↔B: vai trò phần dư chính · A↔C: vai trò phần dư từng tầng
A↔D: vai trò kết nối tắt · A↔E: vai trò tăng độ sâu

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --layer-residual --resample off --weight-cap 3 --ckpt-dir ck_A
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --layer-residual --no-main-residual --resample off --weight-cap 3 --ckpt-dir ck_B
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --resample off --weight-cap 3 --ckpt-dir ck_C
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --layer-residual --skip-mode none --resample off --weight-cap 3 --ckpt-dir ck_D
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 2 --resample off --weight-cap 3 --ckpt-dir ck_E
```
