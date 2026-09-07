"""Lưu / tải cluster graph sau khi chuyển đổi từ packet-level.

VÌ SAO CẦN: mặc định builder tạo graph trong RAM rồi mất khi tắt chương trình.
Module này lưu graph ra đĩa để: (1) không phải build lại mỗi lần (nhanh hơn),
(2) chia sẻ / kiểm tra dữ liệu, (3) tái lập thí nghiệm.

Định dạng: PyTorch .pt (serialize list[Data] của PyG). Kèm 1 file .json metadata
mô tả số graph, số chiều feature, tên các feature — để người khác hiểu dữ liệu.
"""
from __future__ import annotations
import os
import json
import torch

from ..utils.common import get_logger

log = get_logger()

# Tên các đặc trưng — để tài liệu hóa dữ liệu graph
NODE_FEATURE_NAMES = [
    "rank_mean", "dio_cnt", "dao_cnt", "dis_cnt", "pkt_sent", "pkt_recv",
    "interval_mean", "rdao_mean", "rdio_mean", "sdio_mean", "sdao_mean",
    "rank_min", "len_mean",
]
EDGE_FEATURE_NAMES = [
    "pkt_count", "len_mean", "interval_mean",
    "rdao_mean", "rdio_mean", "sdio_mean", "sdao_mean",
    "dio_ratio", "dao_ratio", "is_reverse",
]


def save_graphs(graphs, out_dir: str, name: str = "cluster_graphs",
                class_names=None, window_seconds=None):
    """Lưu list[Data] thành <out_dir>/<name>.pt + metadata .json."""
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, f"{name}.pt")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")

    torch.save(graphs, pt_path)

    n_attack = sum(int((g.y == 1).sum()) for g in graphs)
    n_total = sum(g.y.numel() for g in graphs)
    meta = {
        "num_graphs": len(graphs),
        "total_nodes": int(n_total),
        "attack_nodes": int(n_attack),
        "attack_ratio": round(n_attack / max(1, n_total), 4),
        "n_node_features": int(graphs[0].x.shape[1]) if graphs else 0,
        "n_edge_features": int(graphs[0].edge_attr.shape[1]) if graphs else 0,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "class_names": class_names or ["Normal", "Sinkhole"],
        "window_seconds": window_seconds,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Đã lưu %d graph -> %s", len(graphs), pt_path)
    log.info("Metadata -> %s", meta_path)
    return pt_path, meta_path


def load_graphs(pt_path: str):
    """Tải lại list[Data] đã lưu."""
    graphs = torch.load(pt_path, weights_only=False)
    log.info("Đã tải %d graph từ %s", len(graphs), pt_path)
    return graphs


def inspect_graph(g, idx=0):
    """In chi tiết MỘT graph để hình dung dữ liệu (dùng khi debug/giải thích)."""
    print(f"=== Cluster graph #{idx} ===")
    print(f"  Số node      : {g.x.shape[0]}")
    print(f"  Số cạnh      : {g.edge_index.shape[1]}")
    print(f"  Node features: shape {tuple(g.x.shape)} (mỗi node {g.x.shape[1]} đặc trưng)")
    print(f"  Edge features: shape {tuple(g.edge_attr.shape)}")
    print(f"  Nhãn node    : {g.y.tolist()}")
    print(f"  edge_index   :")
    print(f"    src: {g.edge_index[0].tolist()}")
    print(f"    dst: {g.edge_index[1].tolist()}")
    print(f"  Node feature đầu tiên (node 0):")
    for name, val in zip(NODE_FEATURE_NAMES, g.x[0].tolist()):
        print(f"    {name:16s}: {val:.3f}")
    print(f"  Edge feature đầu tiên (cạnh 0):")
    for name, val in zip(EDGE_FEATURE_NAMES, g.edge_attr[0].tolist()):
        print(f"    {name:16s}: {val:.3f}")
