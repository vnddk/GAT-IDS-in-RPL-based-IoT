"""Baseline tabular: Random Forest & XGBoost cho node classification.

Cách hoạt động: flatten node features từ graph thành bảng phẳng (mỗi dòng = 1 node),
KHÔNG dùng cấu trúc đồ thị. Đây là baseline "fair" để chứng minh giá trị của GNN:
nếu GNN không thắng RF/XGBoost thì cấu trúc đồ thị không giúp gì.

Bổ sung: ngoài node features gốc, còn thêm aggregated neighbor features
(mean, max của neighbors) để cho tabular model "thấy" thông tin lân cận — công bằng hơn.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ..utils.common import get_logger

log = get_logger()


def _flatten_graphs(graphs, add_neighbor_agg=True):
    """Chuyển list[Data] thành (X, y) bảng phẳng cho sklearn.

    Mỗi node → 1 dòng. Features = node features gốc + neighbor aggregation.
    """
    all_x, all_y = [], []
    for g in graphs:
        x = g.x.numpy()                  # [n, d_n]
        y = g.y.numpy()                  # [n]
        n = x.shape[0]

        if add_neighbor_agg and g.edge_index.numel() > 0:
            # Tính mean + max features của neighbors cho mỗi node
            ei = g.edge_index
            neigh_mean = np.zeros_like(x)
            neigh_max = np.full_like(x, -1e9)
            neigh_count = np.zeros(n)
            for e in range(ei.shape[1]):
                src, dst = int(ei[0, e]), int(ei[1, e])
                if dst < n and src < n:
                    neigh_mean[dst] += x[src]
                    neigh_max[dst] = np.maximum(neigh_max[dst], x[src])
                    neigh_count[dst] += 1
            neigh_count = np.maximum(neigh_count, 1).reshape(-1, 1)
            neigh_mean /= neigh_count
            neigh_max = np.where(neigh_max > -1e8, neigh_max, 0)
            x_aug = np.concatenate([x, neigh_mean, neigh_max], axis=1)
        else:
            x_aug = x

        all_x.append(x_aug)
        all_y.append(y)
    return np.concatenate(all_x), np.concatenate(all_y)


def train_eval_rf(train_g, test_g, num_classes, class_names, n_estimators=200):
    """Train Random Forest, eval trên test, trả metrics."""
    from ..utils.metrics import compute_metrics
    log.info("  RF: flattening graphs...")
    X_tr, y_tr = _flatten_graphs(train_g)
    X_te, y_te = _flatten_graphs(test_g)
    log.info("  RF: train %d samples, test %d samples, %d features",
             len(y_tr), len(y_te), X_tr.shape[1])

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    m = compute_metrics(y_te, y_pred, num_classes, class_names)
    return m


def train_eval_xgboost(train_g, test_g, num_classes, class_names, n_estimators=200):
    """Train XGBoost, eval trên test, trả metrics."""
    from ..utils.metrics import compute_metrics
    log.info("  XGBoost: flattening graphs...")
    X_tr, y_tr = _flatten_graphs(train_g)
    X_te, y_te = _flatten_graphs(test_g)
    log.info("  XGBoost: train %d samples, test %d, %d features",
             len(y_tr), len(y_te), X_tr.shape[1])

    # Remap labels to 0..n (XGBoost yêu cầu)
    present = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
    remap = {c: i for i, c in enumerate(present)}
    inv_remap = {i: c for c, i in remap.items()}
    y_tr_r = np.array([remap[c] for c in y_tr])
    y_te_r = np.array([remap[c] for c in y_te])

    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=8, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        tree_method='hist', random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr_r)
    y_pred_r = clf.predict(X_te)
    y_pred = np.array([inv_remap[c] for c in y_pred_r])
    m = compute_metrics(y_te, y_pred, num_classes, class_names)
    return m
