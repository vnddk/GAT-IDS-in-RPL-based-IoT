# Thử nghiệm 3 tầng GAT (3-hop)

## BẰNG CHỨNG CẦN ĐỌC TRƯỚC KHI CHẠY

Tôi đo trực tiếp trên đồ thị RADAR thật (tệp Blackhole/101, 30 cửa sổ):

| Chỉ số | Giá trị |
|---|---|
| Số node trung bình | 18,0 |
| Bậc trung bình | **7,8 láng giềng/node** |
| Đường kính đồ thị | 4,00 hop |
| Bán kính | 2,00 hop |

Vùng phủ theo số hop — tỉ lệ node trong mạng mà một node "nhìn thấy":

| Số hop | % node được phủ |
|---|---|
| 1 | 48,8% |
| **2 (hiện tại)** | **86,4%** |
| **3 (thử nghiệm)** | **95,7%** |
| 4 | 100,0% |

**Ý nghĩa:** đồ thị RADAR RẤT DÀY (bậc 7,8 trên 18 node). Với 2 tầng, mỗi node
đã tổng hợp từ 86% toàn mạng. Với 3 tầng, con số là 96% — nghĩa là gần như MỌI
node đều tổng hợp từ MỌI node khác, nên biểu diễn của chúng hội tụ về nhau.
Đây chính là định nghĩa của hiện tượng làm mịn quá mức (over-smoothing).

Bằng chứng bổ trợ từ chính thí nghiệm trước: ở 2 tầng, lớp worst_parent từng
sụt F1 từ 0,410 xuống 0,154 khi tăng số epoch — dấu hiệu làm mịn quá mức đã
xuất hiện. Kết nối tắt đặc trưng gốc là thứ đã cứu lớp này.

**Dự đoán trung thực:** 3 tầng nhiều khả năng cho macro-F1 THẤP HƠN 2 tầng,
đặc biệt ở các lớp tấn công đơn-node (worst_parent, selective_forward,
blackhole). Nhưng đây vẫn là thí nghiệm ĐÁNG LÀM — một dòng ablation có bằng
chứng số về giới hạn độ sâu là đóng góp hợp lệ cho luận văn.

---

## Đã hiện thực

`num_layers` cấu hình được, tương thích ngược hoàn toàn (L=2 vẫn cho đúng
49.219 tham số như trước).

| L | Số tham số | Ghi chú |
|---|---|---|
| 2 | 49.219 | mặc định hiện tại |
| 3 | 92.996 | +89% tham số |
| 4 | 136.773 | +178% |

Cấu trúc: L−1 tầng đầu dùng `concat=True` (48×3 = 144 chiều), tầng cuối dùng
`concat=False` (32 chiều). Phần dư chính vẫn bắc từ h⁰ tới đầu ra tầng cuối.

Thêm tuỳ chọn `--layer-residual`: bật phần dư TỪNG TẦNG cho các tầng giữa.
Khi tăng độ sâu, đây là biện pháp chuẩn để chống làm mịn quá mức — mỗi tầng
giữ lại biểu diễn của tầng trước thay vì bị trung bình hoá hoàn toàn.

---

## Lệnh chạy

```bash
# Mốc so sánh: 2 tầng (macro-F1 hiện tại 0,8558)
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 2 --ckpt-dir ck_L2

# 3 tầng, không phần dư từng tầng
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --ckpt-dir ck_L3

# 3 tầng, CÓ phần dư từng tầng (biện pháp chống làm mịn quá mức)
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 3 --layer-residual --ckpt-dir ck_L3res

# 4 tầng để thấy xu hướng rõ hơn (tuỳ chọn)
python scripts/train_all.py --epochs 120 --models EdgeGAT --num-layers 4 --layer-residual --ckpt-dir ck_L4

python scripts/test_all.py --ckpt-dir ck_L2
python scripts/test_all.py --ckpt-dir ck_L3
python scripts/test_all.py --ckpt-dir ck_L3res
```

## Cách đọc kết quả

Đừng chỉ nhìn macro-F1 tổng. Hãy so **F1 của bốn lớp tấn công đơn-node**
(blackhole, selective_forward, worst_parent, và local_repair) giữa L=2 và L=3.
Nếu chúng tụt trong khi các lớp khác giữ nguyên, đó là bằng chứng trực tiếp
của làm mịn quá mức — và là một quan sát rất tốt cho Chương 5.

Kèm theo, chạy trực quan hoá để đo định lượng:

```bash
python scripts/tsne_layers.py --ckpt-dir ck_L3 --layers h1,h2,final
```

Nếu điểm silhouette ở tầng cuối THẤP HƠN so với cấu hình L=2, đó là số đo
khách quan cho thấy độ sâu tăng làm giảm khả năng tách cụm.
