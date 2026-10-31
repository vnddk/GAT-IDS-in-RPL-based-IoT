"""Chạy ablation tự động — đúng các so sánh trong đề cương (KC-2).

So sánh:
  A. GAT node-only   (centralized, no edge)
  B. Edge-aware GAT  (centralized, có edge)   <- chứng minh giá trị edge features
  C. Edge-aware GAT  (federated, có edge)     <- hệ thống đề xuất đầy đủ

Chạy:  python scripts/run_ablation.py
"""
from __future__ import annotations
import os
import sys
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.loader import get_datasets, make_loaders
from src.models.edge_gat import build_model
from src.utils.engine import make_criterion, train_centralized
from src.federated.fed import train_federated

log = get_logger()


def run_one(cfg, train_g, val_g, test_g, meta, device, tag):
    set_seed(cfg["seed"])                  # cùng seed để so sánh công bằng
    train_loader, val_loader, test_loader = make_loaders(
        train_g, val_g, test_g, cfg["train"]["batch_size"])
    model = build_model(cfg, meta).to(device)
    criterion = make_criterion(cfg, train_g, meta["num_classes"], device)
    log.info("=== [%s] params=%d ===", tag, model.count_parameters())
    if cfg["federated"]["enabled"]:
        _, m = train_federated(model, train_g, val_loader, test_loader,
                               criterion, cfg, meta, device)
    else:
        _, m = train_centralized(model, (train_loader, val_loader, test_loader),
                                 criterion, cfg, meta, device)
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=None,
                    help="Ghi đè window_seconds cho cả 3 phase")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    base = load_config("configs/default.yaml")
    base["train"]["epochs"] = args.epochs
    if args.window is not None:
        base["data"]["window_seconds"] = args.window
    device = get_device(base["device"])
    set_seed(base["seed"])

    # Dùng CÙNG một split dữ liệu cho cả 3 cấu hình
    train_g, val_g, test_g, meta = get_datasets(base)

    results = {}

    # A. node-only centralized
    cfgA = copy.deepcopy(base)
    cfgA["model"]["use_edge_features"] = False
    cfgA["federated"]["enabled"] = False
    results["A_node_only_centralized"] = run_one(
        cfgA, train_g, val_g, test_g, meta, device, "A node-only centralized")

    # B. edge-aware centralized
    cfgB = copy.deepcopy(base)
    cfgB["model"]["use_edge_features"] = True
    cfgB["federated"]["enabled"] = False
    results["B_edge_aware_centralized"] = run_one(
        cfgB, train_g, val_g, test_g, meta, device, "B edge-aware centralized")

    # C. edge-aware federated
    cfgC = copy.deepcopy(base)
    cfgC["model"]["use_edge_features"] = True
    cfgC["federated"]["enabled"] = True
    cfgC["federated"]["rounds"] = 40
    results["C_edge_aware_federated"] = run_one(
        cfgC, train_g, val_g, test_g, meta, device, "C edge-aware federated")

    # Tổng hợp
    log.info("\n" + "=" * 56)
    log.info("BẢNG ABLATION (macro-F1 trên test)")
    log.info("=" * 56)
    for k, v in results.items():
        log.info("  %-32s macroF1=%.4f acc=%.4f",
                 k, v["macro_f1"], v["accuracy"])

    delta = (results["B_edge_aware_centralized"]["macro_f1"]
             - results["A_node_only_centralized"]["macro_f1"])
    log.info("-" * 56)
    log.info("  Giá trị edge features (B - A): %+.4f macroF1 %s",
             delta, "(KC-2 PASS)" if delta >= 0.02 else "(KC-2 cần xem lại)")

    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Đã lưu -> ablation_results.json")


if __name__ == "__main__":
    main()
