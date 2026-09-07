"""So sánh 6 kiến trúc trên cùng dataset, cùng split, cùng metrics.

Chạy: python scripts/run_comparison.py [--dataset radar] [--data-dir data/radar] [--epochs 60]

6 models:
  1. RF          — Random Forest (tabular, bỏ qua đồ thị)
  2. XGBoost     — XGBoost (tabular, bỏ qua đồ thị)
  3. MLP         — Deep MLP (bỏ qua đồ thị, chỉ node features)
  4. GCN         — Graph Convolutional Network (topology, KHÔNG edge features)
  5. E-GraphSAGE — Edge-GraphSAGE (topology + edge features, KHÔNG attention)
  6. EdgeGAT     — Edge-aware GAT + DyT (topology + edge + attention) ← đề xuất

Kết quả lưu comparison_results.json + in bảng tổng hợp.
"""
from __future__ import annotations
import os
import sys
import json
import copy
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.loader import get_datasets, make_loaders
from src.models.edge_gat import build_model as build_gat
from src.models.gnn_baselines import build_baseline
from src.models.tabular_baselines import train_eval_rf, train_eval_xgboost
from src.utils.engine import make_criterion, train_centralized
from src.utils.metrics import format_report

log = get_logger()


def _run_gnn(name, model, loaders, train_g, cfg, meta, device):
    """Train + eval cho PyTorch model (MLP/GCN/E-GraphSAGE/EdgeGAT)."""
    t0 = time.time()
    criterion = make_criterion(cfg, train_g, meta["num_classes"], device)
    model, test_m = train_centralized(model, loaders, criterion, cfg, meta, device)
    elapsed = time.time() - t0
    test_m["time_seconds"] = round(elapsed, 1)
    test_m["params"] = model.count_parameters()
    return test_m


def main():
    ap = argparse.ArgumentParser(description="So sánh 6 models")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="comparison_results.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["epochs"] = args.epochs
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.data_dir:
        cfg["data"]["csv_dir"] = args.data_dir
    if args.window:
        cfg["data"]["window_seconds"] = args.window

    device = get_device(cfg["device"])

    # ── Tải dữ liệu CÙNG MỘT split cho tất cả models ────────────
    set_seed(cfg["seed"])
    train_g, val_g, test_g, meta = get_datasets(cfg)
    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"])
    nc, cn = meta["num_classes"], meta["class_names"]
    log.info("Dataset: %s | train=%d val=%d test=%d | d_n=%d d_e=%d | %d lớp",
             cfg["data"].get("dataset", "?"), len(train_g), len(val_g), len(test_g),
             meta["n_node_features"], meta["n_edge_features"], nc)

    results = {}

    # ── 1. Random Forest ──────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [1/6] Random Forest")
    t0 = time.time()
    set_seed(42)
    m_rf = train_eval_rf(train_g, test_g, nc, cn)
    m_rf["time_seconds"] = round(time.time() - t0, 1)
    m_rf["params"] = "N/A (tree)"
    results["RF"] = m_rf
    log.info("RF: acc=%.4f macroF1=%.4f", m_rf["accuracy"], m_rf["macro_f1"])

    # ── 2. XGBoost ────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [2/6] XGBoost")
    t0 = time.time()
    set_seed(42)
    m_xgb = train_eval_xgboost(train_g, test_g, nc, cn)
    m_xgb["time_seconds"] = round(time.time() - t0, 1)
    m_xgb["params"] = "N/A (tree)"
    results["XGBoost"] = m_xgb
    log.info("XGBoost: acc=%.4f macroF1=%.4f", m_xgb["accuracy"], m_xgb["macro_f1"])

    # ── 3. Deep MLP ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [3/6] Deep MLP (no graph structure)")
    set_seed(42)
    model = build_baseline("mlp", meta, cfg).to(device)
    log.info("  MLP params: %d", model.count_parameters())
    results["MLP"] = _run_gnn("MLP", model, loaders, train_g, cfg, meta, device)

    # ── 4. GCN ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [4/6] GCN (topology, no edge features)")
    set_seed(42)
    model = build_baseline("gcn", meta, cfg).to(device)
    log.info("  GCN params: %d", model.count_parameters())
    results["GCN"] = _run_gnn("GCN", model, loaders, train_g, cfg, meta, device)

    # ── 5. Edge-GraphSAGE ─────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [5/6] Edge-GraphSAGE (topology + edge, no attention)")
    set_seed(42)
    model = build_baseline("edge_graphsage", meta, cfg).to(device)
    log.info("  E-GraphSAGE params: %d", model.count_parameters())
    results["E-GraphSAGE"] = _run_gnn("E-GraphSAGE", model, loaders, train_g,
                                       cfg, meta, device)

    # ── 6. Edge-aware GAT + DyT (đề xuất) ─────────────────────────
    log.info("\n" + "=" * 60)
    log.info(">>> [6/6] Edge-aware GAT + DyT (ĐỀ XUẤT)")
    set_seed(42)
    model = build_gat(cfg, meta).to(device)
    log.info("  EdgeGAT params: %d", model.count_parameters())
    results["EdgeGAT"] = _run_gnn("EdgeGAT", model, loaders, train_g, cfg, meta, device)

    # ── BẢNG TỔNG HỢP ────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("BẢNG SO SÁNH 6 MODELS (cùng dataset, cùng split)")
    log.info("=" * 70)
    log.info("  %-16s %8s %10s %10s %8s",
             "Model", "Acc", "MacroF1", "Params", "Time(s)")
    log.info("  " + "-" * 56)
    for name, m in results.items():
        log.info("  %-16s %8.4f %10.4f %10s %8s",
                 name, m["accuracy"], m["macro_f1"],
                 str(m.get("params", "?")), str(m.get("time_seconds", "?")))

    # Xếp hạng
    ranked = sorted(results.items(), key=lambda x: -x[1]["macro_f1"])
    log.info("\n  XẾP HẠNG theo Macro-F1:")
    for i, (name, m) in enumerate(ranked, 1):
        marker = " ← ĐỀ XUẤT" if name == "EdgeGAT" else ""
        log.info("    #%d %-16s macroF1=%.4f%s", i, name, m["macro_f1"], marker)

    # ── Per-class F1 cho top-3 ────────────────────────────────────
    log.info("\n  PER-CLASS F1 (3 model tốt nhất):")
    for name, m in ranked[:3]:
        log.info("  --- %s ---", name)
        for c in cn:
            f = m["per_class_f1"].get(c, 0)
            if f > 0:
                log.info("    %-20s %.3f", c, f)

    # ── Lưu ───────────────────────────────────────────────────────
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log.info("\nĐã lưu -> %s", args.out)


if __name__ == "__main__":
    main()
