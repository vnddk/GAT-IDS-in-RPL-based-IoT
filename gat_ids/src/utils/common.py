"""Tiện ích chung: nạp config, cố định seed, chọn device, logger."""
from __future__ import annotations
import os
import random
import logging
from dataclasses import dataclass

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Nạp file YAML thành dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Cố định mọi nguồn ngẫu nhiên để thí nghiệm tái lập được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đảm bảo tái lập (đánh đổi một chút tốc độ)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    """Chọn thiết bị tính toán."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "fededge") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class EarlyStopping:
    """Dừng sớm khi val metric không cải thiện sau `patience` epoch."""
    patience: int = 12
    mode: str = "max"          # max cho F1, min cho loss
    best: float = None
    counter: int = 0
    should_stop: bool = False

    def step(self, value: float) -> bool:
        """Trả về True nếu value là tốt nhất (để lưu checkpoint)."""
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
