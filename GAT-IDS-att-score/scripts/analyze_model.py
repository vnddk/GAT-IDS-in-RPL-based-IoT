"""Demo: kiểm định builder + phân tích model học đặc trưng.

Chạy:  python scripts/analyze_model.py

Gồm 2 phần:
  PHẦN 1 — Kiểm định builder: với mỗi file CSV, so khớp graph vs packet gốc.
  PHẦN 2 — Phân tích học đặc trưng: train nhanh rồi chạy 4 phân tích
           (separability, permutation importance, attention, calibration).
"""
from __future__ import annotations
import os
import sys
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.uos_builder import build_graphs_from_df
from src.data.validate_builder import validate_conversion
from src.data.loader import get_datasets, make_loaders
from src.models.edge_gat import build_model
from src.utils.engine import make_criterion, train_centralized
from src.explain.feature_analysis import run_feature_analysis

log = get_logger()


def part1_validate(cfg):
    log.info("########## PHẦN 1: KIỂM ĐỊNH BUILDER ##########")
    files = sorted(glob.glob(os.path.join(cfg["data"]["csv_dir"], "*.csv")))
    ws = cfg["data"]["window_seconds"]
    any_done = False
    for f in files[:3]:                      # kiểm 3 file đầu cho gọn
        if os.path.getsize(f) < 16:
            continue
        df = pd.read_csv(f)
        if df.shape[0] == 0:
            continue
        graphs = build_graphs_from_df(df, ws)
        log.info(">>> File: %s", os.path.basename(f))
        validate_conversion(df, graphs, ws, verbose=True)
        any_done = True
    if not any_done:
        log.warning("Không có CSV thật trong %s — bỏ qua phần kiểm định.",
                    cfg["data"]["csv_dir"])


def part2_feature_learning(cfg):
    log.info("########## PHẦN 2: PHÂN TÍCH HỌC ĐẶC TRƯNG ##########")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    train_g, val_g, test_g, meta = get_datasets(cfg)
    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"])
    model = build_model(cfg, meta).to(device)
    crit = make_criterion(cfg, train_g, meta["num_classes"], device)
    model, _ = train_centralized(model, loaders, crit, cfg, meta, device)
    # Phân tích trên tập test
    run_feature_analysis(model, loaders[2], device, meta)


def main():
    cfg = load_config("configs/default.yaml")
    cfg["train"]["epochs"] = 50          # gọn cho demo
    part1_validate(cfg)
    part2_feature_learning(cfg)


if __name__ == "__main__":
    main()
