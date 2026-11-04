"""Sinh dữ liệu synthetic dạng cluster-graph cho RPL-IoT.

QUAN TRỌNG: Đây KHÔNG phải noise ngẫu nhiên. Mỗi loại tấn công được nhúng
một "chữ ký" (signature) đặc trưng vào node/edge features, mô phỏng đúng cơ chế
tấn công RPL, để mô hình thực sự có cái để học. Nhờ vậy framework chạy được
end-to-end và cho kết quả có ý nghĩa ngay cả khi chưa có dataset thật.

Mỗi graph = 1 cluster trong 1 cửa sổ thời gian.
  - Node features (d_n=10): [rank, etx_node, pdr_node, rssi_node, dio_cnt,
                             dao_cnt, dis_cnt, dversion, parent_change, sibling]
  - Edge features (d_e=5):  [etx_link, pdr_link, rssi_link, hop, forward_ratio]
  - Nhãn: gán ở mức GRAPH (loại tấn công của cluster) rồi suy ra nhãn node.
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data

# Thứ tự lớp phải khớp configs/default.yaml -> data.classes
CLASS_NAMES = ["Normal", "Sinkhole", "Blackhole", "Flooding", "Version", "Rank"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def _random_tree_edges(n: int, rng: np.random.Generator):
    """Sinh cây DODAG: nút 0 là root, mỗi nút con nối tới 1 parent đã có."""
    src, dst = [], []
    for child in range(1, n):
        parent = rng.integers(0, child)
        # cạnh 2 chiều để message passing đi cả lên lẫn xuống
        src += [child, parent]
        dst += [parent, child]
    if n == 1:                      # graph 1 nút: thêm self-loop
        src, dst = [0], [0]
    return torch.tensor([src, dst], dtype=torch.long)


def _base_features(n: int, rng: np.random.Generator):
    """Đặc trưng nền của mạng BÌNH THƯỜNG (chưa bị tấn công)."""
    node = np.zeros((n, 10), dtype=np.float32)
    node[:, 0] = rng.uniform(128, 512, n)      # rank
    node[:, 1] = rng.uniform(1.0, 1.6, n)      # etx_node
    node[:, 2] = rng.uniform(0.92, 1.0, n)     # pdr_node
    node[:, 3] = rng.uniform(-75, -55, n)      # rssi_node
    node[:, 4] = rng.uniform(8, 16, n)         # dio_cnt
    node[:, 5] = rng.uniform(4, 10, n)         # dao_cnt
    node[:, 6] = rng.uniform(0, 3, n)          # dis_cnt
    node[:, 7] = np.zeros(n)                   # dversion
    node[:, 8] = rng.uniform(0, 1, n)          # parent_change
    node[:, 9] = rng.uniform(0, 4, n)          # sibling

    n_edges_dir = max(1, (n - 1) * 2)
    edge = np.zeros((n_edges_dir, 5), dtype=np.float32)
    edge[:, 0] = rng.uniform(1.0, 1.6, n_edges_dir)    # etx_link
    edge[:, 1] = rng.uniform(0.90, 1.0, n_edges_dir)   # pdr_link
    edge[:, 2] = rng.uniform(-75, -55, n_edges_dir)    # rssi_link
    edge[:, 3] = rng.uniform(1, 5, n_edges_dir)        # hop
    edge[:, 4] = rng.uniform(0.9, 1.0, n_edges_dir)    # forward_ratio
    return node, edge


def _inject_attack(node, edge, label: str, rng: np.random.Generator):
    """Nhúng chữ ký tấn công vào features. Mỗi tấn công đánh vào chỗ khác nhau,
    phản ánh đúng cơ chế thật — đây là điều cho phép mô hình học phân biệt."""
    n = node.shape[0]
    victim = rng.integers(0, n)             # nút bị/đang tấn công

    if label == "Sinkhole":
        # Quảng bá rank rất thấp để hút lưu lượng + ETX giả thấp
        node[victim, 0] = rng.uniform(1, 40)        # rank cực thấp
        node[victim, 1] = rng.uniform(0.3, 0.8)     # etx giả thấp
        node[victim, 5] *= rng.uniform(2, 4)        # dao tăng (hút traffic)
    elif label == "Blackhole":
        # Hút traffic rồi hủy toàn bộ: forward_ratio ~ 0 trên cạnh của victim
        node[victim, 0] = rng.uniform(1, 50)
        for e in range(edge.shape[0]):
            if rng.random() < 0.5:
                edge[e, 4] = rng.uniform(0.0, 0.1)  # forward_ratio ~ 0
                edge[e, 1] = rng.uniform(0.0, 0.2)  # pdr_link sụp
    elif label == "Flooding":
        # DIS flooding: dis_cnt tăng vọt toàn cụm + dio phản hồi tăng
        node[:, 6] += rng.uniform(20, 60, n)        # dis_cnt bùng nổ
        node[:, 4] += rng.uniform(10, 30, n)        # dio phản hồi tăng
    elif label == "Version":
        # Version number attack: dversion > 0, parent_change cao toàn mạng
        node[:, 7] = rng.integers(1, 5, n).astype(np.float32)
        node[:, 8] += rng.uniform(3, 8, n)          # parent_change tăng mạnh
    elif label == "Rank":
        # Decreased rank: victim hạ rank vừa phải (tinh vi hơn sinkhole)
        node[victim, 0] *= rng.uniform(0.2, 0.5)
        node[victim, 8] += rng.uniform(2, 5)        # parent_change quanh victim
    # "Normal": không nhúng gì

    return node, edge, victim


def make_graph(label: str, rng: np.random.Generator, nodes_range=(4, 8)) -> Data:
    """Tạo 1 cluster-graph PyG với nhãn node."""
    n = int(rng.integers(nodes_range[0], nodes_range[1] + 1))
    edge_index = _random_tree_edges(n, rng)
    node, edge = _base_features(n, rng)

    # Đảm bảo số cạnh khớp edge_index
    n_dir = edge_index.shape[1]
    if edge.shape[0] != n_dir:
        edge = np.resize(edge, (n_dir, 5)).astype(np.float32)

    node, edge, victim = _inject_attack(node, edge, label, rng)

    # Nhãn node: tấn công cục bộ (Sinkhole/Blackhole/Rank) chỉ gán victim,
    # tấn công toàn cục (Flooding/Version) gán mọi nút. Normal = 0.
    y = np.zeros(n, dtype=np.int64)
    if label in ("Flooding", "Version"):
        y[:] = CLASS_TO_IDX[label]
    elif label != "Normal":
        y[victim] = CLASS_TO_IDX[label]

    # Chuẩn hóa nhẹ (z-score thô theo cột) — pipeline thật sẽ fit scaler trên train
    x = torch.tensor(node, dtype=torch.float32)
    ea = torch.tensor(edge, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=ea,
                y=torch.tensor(y, dtype=torch.long))


def generate_dataset(n_graphs: int, nodes_range=(4, 8), seed: int = 42):
    """Sinh danh sách Data, phân phối lớp cân bằng tương đối."""
    rng = np.random.default_rng(seed)
    graphs = []
    # Normal chiếm ~40%, 5 tấn công chia đều phần còn lại
    probs = [0.40, 0.12, 0.12, 0.12, 0.12, 0.12]
    for _ in range(n_graphs):
        label = rng.choice(CLASS_NAMES, p=probs)
        graphs.append(make_graph(label, rng, nodes_range))
    return graphs
