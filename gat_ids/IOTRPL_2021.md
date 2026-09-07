# Bộ dữ liệu IoT-RPL 2021 — builder và CẢNH BÁO quan trọng

## Đã hiện thực

`src/data/iotrpl_builder.py` — 16 đặc trưng node, 8 đặc trưng cạnh, 5 lớp
(Normal, Blackhole, Flooding, Version, Rank). Đăng ký `dataset: iotrpl`.

```bash
python scripts/convert_data.py --dataset iotrpl --data-dir data/iotrpl --window 2000
python scripts/train_all.py --epochs 120 --models EdgeGAT
python scripts/test_all.py
```

`--window` ở đây là SỐ DÒNG mỗi đồ thị, không phải giây (xem lý do bên dưới).

---

## Khác biệt cốt lõi so với RADAR

| | RADAR | IoT-RPL 2021 |
|---|---|---|
| Cột thời gian | PHY_LAYER_START_TIME | **KHÔNG CÓ** |
| Node đích | 1 node | **DANH SÁCH** (TB 4,23 node) |
| Chặng vật lý | TRANSMITTER/RECEIVER | **KHÔNG CÓ** |
| RPL_RANK | có | **KHÔNG CÓ** |
| NEXT_HOP_IP | có → suy cha DODAG | **KHÔNG CÓ** |
| Số lớp | 16 | 5 |
| Số node | ~18/đồ thị | 16 cố định |

**Hệ quả 1** — không có thời gian nên phải phân cửa sổ theo **số dòng**
(mặc định 2000 gói/đồ thị) thay vì theo giây.

**Hệ quả 2** — cột `to` là danh sách nên mỗi dòng sinh nhiều cạnh (quảng bá);
chính danh sách này cho tập láng giềng, thay cho TRANSMITTER/RECEIVER.

**Hệ quả 3** — không có rank/version/next-hop nên TOÀN BỘ nhóm đặc trưng
rank của RADAR (`rank_vs_parent`, `diff_rank`, `rank_per_hop`…) không tái tạo
được. Bộ đặc trưng phải xây lại quanh: loại bản tin điều khiển, cấu trúc
quảng bá, và tỉ lệ gửi/nhận.

---

## ⚠⚠ CẢNH BÁO NGHIÊM TRỌNG: NHÃN GẮN CỨNG VỚI NODE ID

Kiểm tra trên 400.000 gói của 0.csv:

| Node ID | Lớp |
|---|---|
| 1–11 | Normal 100% |
| **12** | **Blackhole 100%** |
| **13, 16** | **Flooding 100%** |
| **14** | **Rank 100%** |
| **15** | **Version 100%** |

**16/16 node có 100% gói thuộc đúng MỘT lớp.**

Nghĩa là trong bộ này, kẻ tấn công là các node ID CỐ ĐỊNH suốt toàn bộ mô
phỏng. Bất kỳ đặc trưng nào tương quan với danh tính node đều dự đoán được
nhãn — mô hình có thể chỉ học "node 12 = Blackhole" thay vì học hành vi.

Kiểm toán tự động đã gắn cờ `send_share` với F1 một cột = **0,906**. Đặc trưng
này tính hoàn toàn từ `from`/`to`, KHÔNG chạm cột `label`, nên không phải rò rỉ
nhãn theo nghĩa kỹ thuật. Nhưng nó phản ánh đúng vấn đề trên: node 12 luôn có
tỉ lệ gửi/nhận đặc trưng, và node 12 luôn là Blackhole.

### Hệ quả cho luận văn

Kết quả cao trên bộ này KHÔNG chứng minh mô hình phát hiện được tấn công —
có thể nó chỉ nhận ra node nào là node nào. Ba cách xử lý:

1. **Chia tập theo NODE** (khuyến nghị): huấn luyện trên một số node tấn công,
   kiểm thử trên node tấn công KHÁC. Nhưng bộ này chỉ có 1–2 node mỗi lớp
   tấn công nên gần như không khả thi.

2. **Chia tập theo TỆP**: huấn luyện trên 0–6.csv, kiểm thử trên 7–9.csv.
   Vẫn cùng node ID nhưng khác lần chạy mô phỏng. Đây là phương án thực tế
   nhất — cần kiểm tra xem 10 tệp có phải 10 kịch bản khác nhau hay không.

3. **Báo cáo kèm cảnh báo**: nêu rõ giới hạn này trong phần Hạn chế, và dùng
   bộ IoT-RPL như bằng chứng BỔ TRỢ chứ không phải bằng chứng chính.

**Khuyến nghị của tôi:** dùng RADAR làm bộ chính (16 lớp, có rank, nhãn không
gắn cứng node), IoT-RPL làm bộ thứ hai để kiểm tra tính khả chuyển của khung
phần mềm — và nêu rõ hạn chế trên. Đây vẫn là đóng góp giá trị: nó cho thấy
khung hoạt động trên dữ liệu có cấu trúc hoàn toàn khác, đồng thời thể hiện
năng lực phát hiện vấn đề hợp lệ của cơ chế kiểm toán.

---

## Ba lỗi đã phát hiện và sửa trong quá trình xây dựng

| Lỗi | Triệu chứng | Nguyên nhân | Đã sửa |
|---|---|---|---|
| `fwd_ratio` bùng nổ | max = 1,996 tỉ | `ns/(nr+ε)` khi nr=0 | đổi thành `ns/(ns+nr+ε)` ∈ [0,1] |
| `dio_ratio` chết | toàn 0 | nhận dạng DIO sai cột | dùng `type_cont_messg` (DIO 148k, DAO 53k) |
| `deg_centrality` > 1 | max = 1,231 | cộng peers_out+peers_in đếm trùng | dùng HỢP hai tập |

Sau khi sửa: 16/16 đặc trưng lành mạnh, không NaN/Inf, mọi tỉ lệ trong [0,1].
