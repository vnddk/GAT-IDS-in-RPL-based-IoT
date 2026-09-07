"""BƯỚC 3 — Test TẤT CẢ models, so sánh với nhãn thật (ground truth).

Chạy: python scripts/test_all.py [--data-dir data/processed] [--ckpt-dir checkpoints]

Tải từng model đã train (Bước 2), predict trên test set, SO SÁNH VỚI NHÃN THẬT,
xuất bảng so sánh + per-class F1 + confusion insight.

Output:
  - Bảng xếp hạng 6 models (accuracy, macro-F1, thời gian predict)
  - Per-class F1 cho từng model vs ground truth
  - Chi tiết: với mỗi model, in số mẫu đúng/sai per-class
  - Lưu test_comparison.json
"""
import os, sys, json, argparse, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.utils.common import get_device, get_logger
from src.data.loader import make_loaders
from src.models.tabular_baselines import _flatten_graphs
from src.models.edge_gat import EdgeAwareGAT
from src.models.gnn_baselines import MODEL_REGISTRY
from src.utils.metrics import compute_metrics, collect_predictions, confusion_matrix

log = get_logger()


def _load_data(data_dir):
    test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)
    with open(os.path.join(data_dir, "meta.json")) as f:
        mr = json.load(f)
    meta = {"num_classes": mr["num_classes"], "class_names": mr["class_names"],
            "n_node_features": mr["n_node_features"], "n_edge_features": mr["n_edge_features"]}
    return test, meta


def _test_tabular(name, ckpt_dir, test_g, meta):
    """Test mô hình cây: RF, XGBoost hoặc LightGBM."""
    model_path = os.path.join(ckpt_dir, name, "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    X_te, y_te = _flatten_graphs(test_g)
    t0 = time.time()

    if name in ("XGBoost", "LightGBM"):
        clf, remap = obj["model"], obj["remap"]
        inv_remap = {i: c for c, i in remap.items()}
        y_pred_r = clf.predict(X_te)
        y_pred = np.array([inv_remap.get(c, 0) for c in y_pred_r])
    else:
        clf = obj
        y_pred = clf.predict(X_te)

    t_pred = time.time() - t0
    m = compute_metrics(y_te, y_pred, meta["num_classes"], meta["class_names"])
    m["predict_time"] = round(t_pred, 3)
    m["y_true"] = y_te.tolist()
    m["y_pred"] = y_pred.tolist()
    return m


def _test_gnn(name, ckpt_dir, test_g, meta, device):
    """Test GNN model (MLP/GCN/E-GraphSAGE/EdgeGAT)."""
    d = os.path.join(ckpt_dir, name)
    config_path = os.path.join(d, "model_config.json")
    model_path = os.path.join(d, "best_model.pt")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None

    with open(config_path) as f:
        mc = json.load(f)

    # Build model theo type — ánh xạ tên linh hoạt
    _TYPE_MAP = {
        "mlp": "mlp", "deepmlp": "mlp",
        "gcn": "gcn", "gcnbaseline": "gcn",
        "e-graphsage": "edge_graphsage", "egraphsage": "edge_graphsage",
        "edge_graphsage": "edge_graphsage", "edgegraphsage": "edge_graphsage",
    }
    mt = mc.get("model_type", name)
    mt_lower = mt.lower().replace(" ", "")

    if mt_lower in ("ne-gat", "negat", "ne_gat"):
        from src.models.ne_gat import NEGAT
        model = NEGAT(
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            num_classes=mc["num_classes"],
            hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
            edge_latent=mc.get("edge_latent", 24),
            node_heads=mc.get("node_heads", 2),
            num_layers=mc.get("num_layers", 3),
            dropout=mc["dropout"],
            classifier_type=mc.get("classifier_type", "mlp"),
            clf_hidden=mc.get("clf_hidden", 128),
            edge_score_mode=mc.get("edge_score_mode", "full"),
            edge_agg=mc.get("edge_agg", "mean_deg"),
            edge_update=mc.get("edge_update", True),
            use_dyt=mc.get("use_dyt", True), verbose=False)
    elif mt_lower in ("edgegat", "edge_gat", "edgeawaredgat"):
        model = EdgeAwareGAT(
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
            num_classes=mc["num_classes"], heads=mc["heads"],
            dropout=mc["dropout"], use_edge_features=mc["use_edge_features"],
            use_dyt=mc.get("use_dyt", True),
            use_input_skip=mc.get("use_input_skip", True),
            skip_mode=mc.get("skip_mode", "raw"),
            num_layers=mc.get("num_layers", 2),
            layer_residual=mc.get("layer_residual", False),
        use_main_residual=mc.get("use_main_residual", True),
        classifier_type=mc.get("classifier_type", "linear"),
        clf_hidden=mc.get("clf_hidden", 64))
    elif mt_lower in _TYPE_MAP:
        reg_key = _TYPE_MAP[mt_lower]
        model = MODEL_REGISTRY[reg_key](
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
            num_classes=mc["num_classes"], dropout=mc["dropout"])
    else:
        log.warning("  Không nhận diện model type: %s", mt)
        return None

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    from torch_geometric.loader import DataLoader as PyGLoader
    loader = PyGLoader(test_g, batch_size=64, shuffle=False)
    t0 = time.time()
    y_true, y_pred = collect_predictions(model, loader, device)
    t_pred = time.time() - t0

    m = compute_metrics(y_true, y_pred, meta["num_classes"], meta["class_names"])
    m["predict_time"] = round(t_pred, 3)
    m["params"] = model.count_parameters()
    m["y_true"] = y_true.tolist()
    m["y_pred"] = y_pred.tolist()
    return m


def _ground_truth_comparison(m, cn):
    """So sánh prediction với ground truth, in chi tiết đúng/sai per-class."""
    y_true = np.array(m["y_true"])
    y_pred = np.array(m["y_pred"])
    lines = []
    for i, c in enumerate(cn):
        mask = y_true == i
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((y_pred[mask] == i).sum())
        wrong = total - correct
        acc = correct / total if total > 0 else 0
        f1 = m["per_class_f1"].get(c, 0)
        lines.append(f"    {c:<22} total={total:>6}  correct={correct:>6}  wrong={wrong:>5}  acc={acc:.3f}  F1={f1:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Bước 3: Test tất cả + so sánh ground truth")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--binary", action="store_true",
                    help="In thêm bảng đánh giá NHỊ PHÂN (tấn công / bình thường) "
                         "cho mọi mô hình. Chỉ số nhị phân LUÔN cao hơn macro-F1 "
                         "đa lớp vì phép gộp xoá lỗi nhầm giữa các loại tấn công.")
    ap.add_argument("--out", default="test_comparison.json")
    args = ap.parse_args()
    device = get_device("cpu")

    test_g, meta = _load_data(args.data_dir)
    cn = meta["class_names"]
    nc = meta["num_classes"]
    log.info("Test set: %d graph | %d lớp", len(test_g), nc)

    model_names = ["RF", "XGBoost", "LightGBM", "MLP", "GCN", "E-GraphSAGE", "EdgeGAT", "NE-GAT"]
    results = {}

    for name in model_names:
        log.info("\n" + "=" * 60)
        log.info(">>> Testing: %s", name)

        if name in ("RF", "XGBoost", "LightGBM"):
            m = _test_tabular(name, args.ckpt_dir, test_g, meta)
        else:
            m = _test_gnn(name, args.ckpt_dir, test_g, meta, device)

        if m is None:
            log.warning("  Bỏ qua %s (chưa train hoặc thiếu file)", name)
            continue

        results[name] = m
        log.info("  Accuracy: %.4f | Macro-F1: %.4f | Predict time: %.3fs",
                 m["accuracy"], m["macro_f1"], m["predict_time"])
        log.info("  SO SÁNH VỚI NHÃN THẬT (ground truth):")
        log.info("\n%s", _ground_truth_comparison(m, cn))
        if getattr(args, "binary", False) and "y_true" in m and "y_pred" in m:
            from src.utils.binary_eval import binary_metrics, format_binary
            bm = binary_metrics(m["y_true"], m["y_pred"])
            m["binary"] = bm
            log.info("\n%s", format_binary(bm))

    # ── BẢNG TỔNG HỢP ────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    # SỬA: số 6 bị viết cứng — bảng in ra 7 dòng vẫn ghi "6 MODELS".
    log.info("BẢNG SO SÁNH %d MODELS — GROUND TRUTH VERIFICATION",
             sum(1 for n in model_names if results.get(n)))
    log.info("=" * 70)
    log.info("  %-16s %8s %10s %10s %10s",
             "Model", "Acc", "MacroF1", "Params", "Pred(s)")
    log.info("  " + "-" * 58)

    for name in model_names:
        m = results.get(name)
        if m is None: continue
        params = m.get("params", "tree")
        log.info("  %-16s %8.4f %10.4f %10s %10.3f",
                 name, m["accuracy"], m["macro_f1"], str(params), m["predict_time"])

    ranked = sorted(results.items(), key=lambda x: -x[1]["macro_f1"])
    log.info("\n  XẾP HẠNG:")
    for i, (name, m) in enumerate(ranked, 1):
        tag = " ← ĐỀ XUẤT" if name == "EdgeGAT" else ""
        log.info("    #%d %-16s macroF1=%.4f%s", i, name, m["macro_f1"], tag)

    # ── GROUND TRUTH CHI TIẾT cho #1 ──────────────────────────────
    if ranked:
        best_name, best_m = ranked[0]
        total_true = len(best_m["y_true"])
        total_correct = sum(1 for t, p in zip(best_m["y_true"], best_m["y_pred"]) if t == p)
        total_wrong = total_true - total_correct
        log.info("\n  GROUND TRUTH tổng (#1 %s):", best_name)
        log.info("    Tổng node test: %d | Đúng: %d | Sai: %d | Accuracy: %.4f",
                 total_true, total_correct, total_wrong, total_correct / total_true)

    # ── Lưu ───────────────────────────────────────────────────────
    save_results = {}
    for name, m in results.items():
        save_m = {k: v for k, v in m.items() if k not in ("y_true", "y_pred")}
        save_results[name] = save_m
    with open(args.out, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    log.info("\nĐã lưu -> %s", args.out)
    # Lưu thêm bản sao trong thư mục điểm kiểm tra để scripts/aggregate_seeds.py
    # gom được kết quả nhiều hạt giống mà không cần đường dẫn thủ công.
    try:
        side = os.path.join(args.ckpt_dir, "test_comparison.json")
        with open(side, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
        log.info("Đã lưu bản sao -> %s", side)
    except Exception as e:
        log.warning("Không lưu được bản sao trong ckpt-dir: %s", e)
    log.info("═══ BƯỚC 3 HOÀN TẤT ═══")


if __name__ == "__main__":
    main()
