"""Điểm vào chính của FedEdge-GAT-IDS.

Chạy:
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --federated      # bật FL
    python main.py --config configs/default.yaml --no-edge        # ablation tắt edge

Pipeline: nạp config -> dữ liệu -> model -> train (centralized/federated)
          -> test -> XAI -> lưu kết quả JSON.
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.loader import get_datasets, make_loaders
from src.models.edge_gat import build_model
from src.utils.engine import make_criterion, train_centralized
from src.federated.fed import train_federated
from src.explain.xai import run_explanations

log = get_logger()


def parse_args():
    ap = argparse.ArgumentParser(description="FedEdge-GAT-IDS")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--federated", action="store_true", help="Bật federated")
    ap.add_argument("--no-edge", action="store_true", help="Ablation: tắt edge features")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window", type=float, default=None,
                    help="Ghi đè window_seconds (vd: 1, 3, 5) để thử nhanh")
    ap.add_argument("--dataset", default=None, choices=["uos", "radar", "synthetic"],
                    help="Chuyển dataset (uos/radar/synthetic)")
    ap.add_argument("--processed-dir", default="data/processed",
                    help="Dùng graph đã convert (Bước 1) nếu có -> bỏ qua đọc CSV")
    ap.add_argument("--data-dir", default=None,
                    help="Thư mục chứa dữ liệu CSV (ghi đè csv_dir)")
    ap.add_argument("--out", default="results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Override bằng cờ dòng lệnh
    if args.federated:
        cfg["federated"]["enabled"] = True
    if args.no_edge:
        cfg["model"]["use_edge_features"] = False
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.window is not None:
        cfg["data"]["window_seconds"] = args.window
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
        if args.dataset == "synthetic":
            cfg["data"]["source"] = "synthetic"
    if args.data_dir is not None:
        cfg["data"]["csv_dir"] = args.data_dir

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    log.info("Device: %s | Federated: %s | Edge features: %s",
             device, cfg["federated"]["enabled"], cfg["model"]["use_edge_features"])

    # 1) Dữ liệu — ưu tiên graph đã convert ở Bước 1 (nhanh hơn đọc CSV rất nhiều)
    import os as _os, json as _json
    _pt = _os.path.join(args.processed_dir, "train.pt")
    if _os.path.exists(_pt):
        log.info("Nạp graph đã convert từ %s (bỏ qua đọc CSV)", args.processed_dir)
        train_g = torch.load(_pt, weights_only=False)
        val_g = torch.load(_os.path.join(args.processed_dir, "val.pt"), weights_only=False)
        test_g = torch.load(_os.path.join(args.processed_dir, "test.pt"), weights_only=False)
        with open(_os.path.join(args.processed_dir, "meta.json")) as f:
            _mr = _json.load(f)
        meta = {"num_classes": _mr["num_classes"], "class_names": _mr["class_names"],
                "n_node_features": _mr["n_node_features"],
                "n_edge_features": _mr["n_edge_features"]}
    else:
        train_g, val_g, test_g, meta = get_datasets(cfg)
    train_loader, val_loader, test_loader = make_loaders(
        train_g, val_g, test_g, cfg["train"]["batch_size"])

    # 2) Model
    model = build_model(cfg, meta).to(device)
    log.info("Model Edge-aware GAT | tham số: %d | edge_dim=%s",
             model.count_parameters(),
             meta["n_edge_features"] if cfg["model"]["use_edge_features"] else "OFF")

    # 3) Loss
    criterion = make_criterion(cfg, train_g, meta["num_classes"], device)

    # 4) Train + Test
    if cfg["federated"]["enabled"]:
        model, test_m = train_federated(
            model, train_g, val_loader, test_loader,
            criterion, cfg, meta, device)
    else:
        model, test_m = train_centralized(
            model, (train_loader, val_loader, test_loader),
            criterion, cfg, meta, device)

    # 5) XAI
    xai = run_explanations(model, test_g, device, cfg)

    # 6) Lưu kết quả
    result = {
        "config": {
            "federated": cfg["federated"]["enabled"],
            "use_edge_features": cfg["model"]["use_edge_features"],
            "aggregator": cfg["federated"]["aggregator"],
            "partition": cfg["federated"]["partition"],
        },
        "test_metrics": test_m,
        "xai": xai,
        "model_params": model.count_parameters(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Đã lưu kết quả -> %s", args.out)


if __name__ == "__main__":
    main()
