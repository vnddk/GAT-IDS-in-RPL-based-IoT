"""Smoke test: kiểm tra pipeline chạy được đầu-cuối với cấu hình nhỏ.

Chạy:  python -m pytest tests/ -v    (hoặc)    python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import set_seed, get_device
from src.data.loader import get_datasets, make_loaders
from src.models.edge_gat import build_model
from src.utils.engine import make_criterion, train_centralized


def _tiny_cfg():
    return {
        "seed": 0, "device": "cpu",
        "data": {"source": "synthetic", "csv_dir": "./data/raw",
                 "window_seconds": 60,
                 "synthetic": {"n_graphs": 120, "nodes_per_graph": [4, 6],
                               "n_node_features": 10, "n_edge_features": 5},
                 "classes": ["Normal", "Sinkhole", "Blackhole", "Flooding", "Version", "Rank"],
                 "split": [0.7, 0.15, 0.15]},
        "model": {"hidden_dim": 16, "out_dim": 16, "heads": 2, "dropout": 0.3,
                  "use_edge_features": True, "edge_dim": 5},
        "train": {"epochs": 5, "lr": 0.01, "weight_decay": 0.0005,
                  "batch_size": 16, "class_weighted_loss": True,
                  "early_stopping_patience": 10, "ckpt_dir": "/tmp/ckpt_smoke"},
        "federated": {"enabled": False, "num_clients": 3, "rounds": 3,
                      "local_epochs": 2, "partition": "iid",
                      "dirichlet_alpha": 0.5, "aggregator": "fedavg", "fedprox_mu": 0.01},
        "explain": {"enabled": False, "gnnexplainer_epochs": 10, "top_k_edges": 5},
    }


def test_pipeline_runs():
    cfg = _tiny_cfg()
    set_seed(cfg["seed"])
    device = get_device("cpu")
    train_g, val_g, test_g, meta = get_datasets(cfg)
    assert len(train_g) > 0 and len(test_g) > 0
    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"])
    model = build_model(cfg, meta).to(device)
    assert model.count_parameters() > 0
    crit = make_criterion(cfg, train_g, meta["num_classes"], device)
    _, test_m = train_centralized(model, loaders, crit, cfg, meta, device)
    assert 0.0 <= test_m["macro_f1"] <= 1.0
    print("OK — macro_f1 =", round(test_m["macro_f1"], 4))


if __name__ == "__main__":
    test_pipeline_runs()
