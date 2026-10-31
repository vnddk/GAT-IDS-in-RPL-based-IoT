# FedEdge-GAT-IDS v3-clean — Đặc trưng SẠCH (không rò rỉ PCKT_LABEL)

Xây lại từ framework GỐC (bản upload đầu đoạn chat) + Cải tiến v3, thay bộ
đặc trưng rò rỉ bằng bộ đặc trưng đúng bản chất DODAG.

## Thay đổi cốt lõi
- radar_builder.py: 13→21 đặc trưng node, 10→10 đặc trưng cạnh, ĐỀU SẠCH.
  Bỏ lab_sent/lab_recv/atk_ratio (suy từ PCKT_LABEL). Thêm nhóm A/B/C theo
  triết lý "độ lệch tự-khai vs hàng-xóm-quan-sát". Dùng TRANSMITTER/RECEIVER
  (láng giềng 1-hop) và NEXT_HOP_IP (cha DODAG). Chi tiết: DacTrung_RPL_Graph.docx
- PCKT_LABEL: CHỈ dùng sinh nhãn y, KHÔNG vào đặc trưng x.

## Cải tiến v3 giữ lại
- Yeo-Johnson (transforms.py), Class-Balanced Focal (losses.py),
  huấn luyện ổn định AdamW+cosine+EMA+grad-clip (engine.py).
- raw-skip CHỈ cho EdgeGAT (đề xuất). GCN/E-GraphSAGE/MLP KHÔNG có raw-skip.
- Sửa phân tầng chia tập (lớp version không còn biến mất khỏi test).
- Kiểm toán rò rỉ: cờ = phép thử gốc-cây 1 cột F1 > 0.90 (bỏ dương-tính-giả theo nMI).

## Chạy
    pip install -r requirements.txt
    python scripts/convert_data.py --dataset radar --data-dir data/radar --window 5
    python scripts/train_all.py --epochs 120
    python scripts/test_all.py

Log convert cần thấy: d_n=21 d_e=10 | kiểm toán "Không phát hiện rò rỉ" |
bảng phân bố node đủ 16 lớp ở cột test.
