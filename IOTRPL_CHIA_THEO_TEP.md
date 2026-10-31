# IoT-RPL: chia tập theo TỆP — kiểm chứng và hiện thực

## Câu hỏi: 10 tệp có phải 10 kịch bản khác nhau?

**Trả lời: KHÔNG.** Chúng là 10 LẦN CHẠY của cùng một cấu hình mạng.

### Bằng chứng

| Kiểm tra | Kết quả |
|---|---|
| md5 toàn tệp | 10 giá trị khác nhau — không phải bản sao y hệt |
| md5 200.000 dòng đầu | tệp **1,2,3,5,6,8,9 GIỐNG HỆT** (cùng mã `6e06489a6ed4`) |
| | chỉ tệp 0 và 7 khác từ đầu |
| Điểm phân kỳ 1.csv ↔ 2.csv | **dòng 260.144** (25,6% tệp) |
| Sau điểm phân kỳ | chỉ 12,2% dòng còn trùng vị trí |
| Node tấn công | **12,13,14,15,16 ở MỌI tệp** |

Diễn giải: cùng cấu hình mô phỏng, cùng hạt giống ở pha khởi động, sau đó
tính ngẫu nhiên mới làm chúng phân kỳ.

---

## Hệ quả và cách xử lý

### ⚠ Vấn đề phát hiện: 260k dòng đầu giống hệt nhau

Nếu train chứa tệp 1–6 và test chứa tệp 8–9, mà các tệp này có 260 nghìn dòng
đầu giống hệt, thì cửa sổ đầu của tệp test sẽ TRÙNG KHÍT với cửa sổ đầu của
tệp train. Đây là rò rỉ train→test thật sự.

**Đã xử lý:** tham số `--skip-head` (mặc định **300.000**) bỏ phần đầu chung
của mỗi tệp trước khi tạo cửa sổ.

### Kiểm chứng sau khi áp dụng

```
train  1100 đồ thị | tệp nguồn: ['0', '1', '2']
val     361 đồ thị | tệp nguồn: ['7']
test    719 đồ thị | tệp nguồn: ['8', '9']

Giao train∩test: {}  — không tệp nào xuất hiện ở hai tập
```

Còn 50/690 đồ thị trùng nội dung giữa train và test (7,2%). Tôi kiểm tra xem
đây có phải rò rỉ không:

| | Tỉ lệ trùng |
|---|---|
| Nội bộ train | 4,1% |
| Nội bộ test | 4,0% |
| train ∩ test | 7,2% |

Trùng lặp xuất hiện **cả trong nội bộ từng tập**, nên đây là ĐẶC TÍNH VỐN CÓ
của dữ liệu: mạng chỉ 16 node với số loại bản tin hạn chế, nhiều cửa sổ 2000
gói cho ra cùng một vector thống kê. Không phải lỗi cách chia tập.

---

## Lệnh chạy

```bash
# Chia theo tệp: train 0-4 | val 5,6 | test 7,8,9
python scripts/convert_data.py --dataset iotrpl --data-dir data/iotrpl \
    --window 2000 --split-by-file "0,1,2,3,4:5,6:7,8,9" --skip-head 300000

python scripts/train_all.py --epochs 120 --models EdgeGAT --resample off --weight-cap 3
python scripts/test_all.py
```

Cú pháp `--split-by-file "train:val:test"`, mỗi phần là danh sách tên tệp
(không đuôi) cách nhau bởi dấu phẩy.

Để so sánh với chia ngẫu nhiên (chứng minh chia theo tệp khắt khe hơn):

```bash
python scripts/convert_data.py --dataset iotrpl --data-dir data/iotrpl \
    --window 2000 --skip-head 300000 --out-dir data/iotrpl_random
```

---

## ⚠ GIỚI HẠN CÒN LẠI — phải nêu trong luận văn

Chia theo tệp **loại bỏ được** rò rỉ do trùng cửa sổ giữa train và test.

Nhưng nó **KHÔNG loại bỏ được** vấn đề gốc: node 12 là Blackhole ở MỌI tệp,
node 13,16 là Flooding, node 14 là Rank, node 15 là Version. Mô hình vẫn có
thể học "node 12 = Blackhole" thay vì học hành vi tấn công.

Kiểm toán vẫn gắn cờ `send_share` với F1 một cột = 0,906 — đặc trưng này tính
hoàn toàn từ `from`/`to`, không chạm cột `label`, nên không phải rò rỉ nhãn
theo nghĩa kỹ thuật. Nhưng nó phản ánh đúng vấn đề trên.

### Cách trình bày trung thực trong luận văn

1. Dùng **RADAR làm bộ chính** (16 lớp, có rank, nhãn không gắn cứng node).
2. Dùng **IoT-RPL làm bộ thứ hai** để kiểm chứng tính khả chuyển của khung
   phần mềm sang dữ liệu có cấu trúc hoàn toàn khác (không thời gian, không
   rank, đích là danh sách quảng bá).
3. Nêu rõ trong phần Hạn chế: kết quả trên IoT-RPL không chứng minh năng lực
   phát hiện tấn công, vì thiết kế mô phỏng gắn cứng danh tính node tấn công.
4. Trình bày việc **phát hiện ra hạn chế này** như một đóng góp — nó cho thấy
   cơ chế kiểm toán tự động hoạt động đúng trên cả bộ dữ liệu mới.

Đây là cách trình bày mạnh hơn nhiều so với báo cáo một con số cao mà không
kiểm tra tính hợp lệ.
