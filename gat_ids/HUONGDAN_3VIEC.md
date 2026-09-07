# Ba việc: ma trận nhầm lẫn · MLP 2 tầng · nhiều hạt giống

## 1. MLP hai tầng ẩn — đã cập nhật

`edge_gat.py` giờ khớp cấu trúc anh dùng:

    Linear(clf_in → 128) → ReLU → Dropout(0,3)
    Linear(128 → 64)     → ReLU → Dropout(0,3)
    Linear(64 → 16)

Tầng thứ hai tự đặt bằng `clf_hidden // 2` (thiết kế hình phễu). `clf_hidden`
mặc định đổi từ 64 lên **128** cho khớp cấu hình anh đã chạy.

```bash
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
    --layer-residual --skip-mode none --no-main-residual --resample off \
    --weight-cap 3 --classifier mlp --ckpt-dir ck_mlp
```

Chỉnh bề rộng bằng `--clf-hidden 256` (tầng 2 sẽ là 128).

---

## 2. Ma trận nhầm lẫn — script mới

```bash
python scripts/confusion_matrix.py --ckpt-dir ck_L3_nomainres
```

Sinh ra ba thứ:
- **Hình** ma trận chuẩn hoá theo hàng (mỗi hàng cộng = 100%), ô đường chéo là recall
- **Bảng TOP cặp nhầm lẫn** in ra màn hình, sắp theo tỉ lệ
- **Tệp CSV** ma trận thô để đưa vào phụ lục luận văn

Phần quan trọng nhất là bảng "ĐỘ PHÂN TÁN CỦA TỪNG LỚP THẬT" — nó cho biết mỗi
lớp bị nhầm sang BAO NHIÊU lớp khác và lớp nào hút nhiều nhất. Đây là thứ hình
t-SNE không trả lời được.

Câu hỏi cần trả lời cho lớp `rank` (recall 0,412): 59% node còn lại bị gán sang
lớp nào? Nếu tập trung vào một lớp cụ thể (ví dụ `sinkhole`) thì hướng cải tiến
là thêm đặc trưng phân biệt hai lớp đó. Nếu rải đều nhiều lớp thì `rank` là lớp
đa modal và cần hướng khác (phân loại hai tầng).

---

## 3. Nhiều hạt giống — đã sửa lỗi và thêm script

### Lỗi đã sửa
`train_all.py` trước đây **hard-code `set_seed(42)`** ở sáu chỗ, nên cờ `--seed`
không có tác dụng. Nay đã truyền đúng hạt giống từ dòng lệnh.

Ngoài ra `test_all.py` giờ lưu thêm bản sao `test_comparison.json` vào chính
thư mục điểm kiểm tra, để script tổng hợp tìm được mà không cần đường dẫn thủ công.

### Quy trình

```bash
# Cấu hình A — có phần dư chính + raw-skip
for s in 42 43 44; do
  python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
      --layer-residual --resample off --weight-cap 3 --seed $s --ckpt-dir ck_A_s$s
  python scripts/test_all.py --ckpt-dir ck_A_s$s
done

# Cấu hình D — không phần dư, không raw-skip
for s in 42 43 44; do
  python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 \
      --layer-residual --skip-mode none --no-main-residual --resample off \
      --weight-cap 3 --seed $s --ckpt-dir ck_D_s$s
  python scripts/test_all.py --ckpt-dir ck_D_s$s
done

# Tổng hợp và so sánh thống kê
python scripts/aggregate_seeds.py \
    --dirs ck_A_s42,ck_A_s43,ck_A_s44 --name A \
    --compare-dirs ck_D_s42,ck_D_s43,ck_D_s44 --compare-name D
```

Windows CMD dùng `for %s in (42 43 44) do ...`

### Kết quả sẽ có

- macro-F1 và accuracy dạng **trung bình ± độ lệch chuẩn**
- **Khoảng tin cậy 95%** (dùng phân phối t, phù hợp n nhỏ)
- F1 từng lớp kèm độ biến động, đánh dấu lớp nào rất nhạy với hạt giống
- **Kết luận thống kê**: hai khoảng tin cậy có chồng nhau không

### Cách diễn giải

Nếu hai khoảng **chồng nhau** — rất có thể, vì sáu cấu hình chỉ cách nhau
0,0086 — thì kết luận đúng là *"các cấu hình tương đương về mặt thống kê"*.
Khi đó chọn cấu hình theo tiêu chí khác: số tham số ít nhất, độ tách cụm
(silhouette) cao nhất, hoặc kiến trúc đơn giản nhất.

Đây là kết luận MẠNH cho luận văn, không phải kết quả thất bại: nó cho thấy
mô hình bền vững với các lựa chọn kiến trúc, và cho phép chọn bản gọn nhất
mà không mất hiệu năng.

Trong luận văn nên viết: *"macro-F1 = 0,8621 ± 0,0xxx (n = 3 hạt giống)"*
thay vì một con số đơn lẻ.
