"""Chuyển CSV packet-level -> cluster graph và LƯU ra đĩa.

Chạy:
    python scripts/export_graphs.py                 # window theo config
    python scripts/export_graphs.py --window 1      # window tùy chọn

Kết quả lưu ở:  data/processed/cluster_graphs.pt        (dữ liệu graph)
                data/processed/cluster_graphs.meta.json (mô tả đặc trưng)
"""
from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import load_config, get_logger
from src.data.uos_builder import load_uos_csv, UOS_CLASSES
from src.data.graph_io import save_graphs, inspect_graph

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml")
    ws = args.window if args.window is not None else cfg["data"]["window_seconds"]

    log.info("Chuyển CSV -> graph với window=%.1fs", ws)
    graphs = load_uos_csv(cfg["data"]["csv_dir"], window_seconds=ws)
    if not graphs:
        log.error("Không có graph nào. Kiểm tra data/raw/ có CSV chưa.")
        return

    # Lưu ra đĩa
    save_graphs(graphs, args.out_dir, name="cluster_graphs",
                class_names=UOS_CLASSES, window_seconds=ws)

    # In thử 1 graph để hình dung
    log.info("--- Ví dụ một graph ---")
    inspect_graph(graphs[0], idx=0)


if __name__ == "__main__":
    main()
