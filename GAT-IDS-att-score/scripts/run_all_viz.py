#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chạy t-SNE và ma trận nhầm lẫn cho MỌI mô hình trên MỘT bộ dữ liệu.

QUY TẮC t-SNE (theo yêu cầu đề tài):
  · EdgeGAT-IDS và NE-GAT-IDS : vẽ t-SNE cho TỪNG TẦNG truyền thông điệp,
    để thấy biểu diễn tách dần qua các tầng.
  · Mọi mô hình còn lại        : chỉ vẽ t-SNE cho TẦNG CUỐI (biểu diễn ngay
    trước bộ phân loại).
  · Mô hình CÂY (RF, XGBoost, LightGBM) không có tầng ẩn. Thay vì bỏ qua,
    script dùng VÉC-TƠ XÁC SUẤT LỚP làm "biểu diễn cuối" — đó đúng là thứ
    cây sinh ra ngay trước phép argmax. Hình sẽ ghi rõ điều này để không
    gây hiểu nhầm rằng cây có không gian nhúng như mạng nơ-ron.

MA TRẬN NHẦM LẪN: vẽ cho TẤT CẢ mô hình, không phân biệt loại.

Chạy một bộ:
    python scripts/run_all_viz.py --data-dir data/radar_chrono \\
           --ckpt-dir checkpoints --out-dir viz/radar

Chạy cả bốn bộ: xem scripts/run_all_viz_4ds.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.common import get_logger, get_device, set_seed
from src.utils.metrics import confusion_matrix as cm_fn

log = get_logger()

# Thứ tự hiển thị. Hai mô hình đề xuất để cuối cho dễ so sánh.
ALL_MODELS = ["RF", "XGBoost", "LightGBM", "MLP", "GCN", "E-GraphSAGE",
              "GAT-Transformer", "EdgeGAT", "NE-GAT"]
PER_LAYER = {"EdgeGAT", "NE-GAT"}        # hai mô hình vẽ t-SNE từng tầng
TREE = {"RF", "XGBoost", "LightGBM"}


# ════════════════════════════ nạp mô hình ════════════════════════════
def _load_tree(name, ckpt_dir):
    """Nạp mô hình cây. Trả (clf, inv_remap) hoặc None.

    SỬA LỖI v3.41: train_all.py lưu XGBoost và LightGBM dưới dạng
    `{"model": clf, "remap": remap}` vì hai thư viện này đòi nhãn liên tục
    0..K-1, còn RF thì lưu trực tiếp bộ phân loại. Bản cũ ở đây trả nguyên
    cái dict nên `_tree_forward` gọi `.predict` trên dict và sinh
    AttributeError; try/except ở vòng ngoài nuốt lỗi và chỉ in "bỏ qua",
    khiến cả ba mô hình cây KHÔNG có ma trận nhầm lẫn lẫn t-SNE. Logic
    dưới đây lấy đúng theo scripts/test_all.py::_test_tabular.
    """
    import joblib
    obj = None
    for fn in (f"{name}.joblib", f"{name.lower()}.joblib", "model.joblib",
               "model.pkl"):
        p = os.path.join(ckpt_dir, name, fn)
        if os.path.exists(p):
            obj = joblib.load(p)
            break
    if obj is None:
        return None
    if isinstance(obj, dict) and "model" in obj:
        remap = obj.get("remap") or {}
        inv = {i: c for c, i in remap.items()}
        return obj["model"], inv
    return obj, None


def _load_gnn(name, ckpt_dir, device, override_dir=None):
    """Dựng lại mô hình nơ-ron từ model_config.json. Trả (model, mc) hoặc None."""
    d = override_dir or os.path.join(ckpt_dir, name)
    cfg_p = os.path.join(d, "model_config.json")
    w_p = os.path.join(d, "best_model.pt")
    if not (os.path.exists(cfg_p) and os.path.exists(w_p)):
        return None
    mc = json.load(open(cfg_p, encoding="utf-8"))
    mt = str(mc.get("model_type", name)).lower().replace(" ", "")

    if mt in ("ne-gat", "negat", "ne_gat"):
        from src.models.ne_gat import NEGAT
        model = NEGAT(
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            num_classes=mc["num_classes"], hidden_dim=mc["hidden_dim"],
            out_dim=mc["out_dim"], edge_latent=mc.get("edge_latent", 24),
            node_heads=mc.get("node_heads", 2),
            num_layers=mc.get("num_layers", 3), dropout=mc["dropout"],
            classifier_type=mc.get("classifier_type", "mlp"),
            clf_hidden=mc.get("clf_hidden", 128),
            edge_score_mode=mc.get("edge_score_mode", "full"),
            edge_agg=mc.get("edge_agg", "mean_deg"),
            edge_update=mc.get("edge_update", True),
            use_dyt=mc.get("use_dyt", True), verbose=False)
    elif mt in ("edgegat", "edge_gat", "edgeawaredgat"):
        from src.models.edge_gat import EdgeAwareGAT
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
    elif mt.startswith("gat-transformer") or mt in ("gattransformer", "gtids"):
        if mc.get("stage") == "ssl":
            log.warning("  %s là checkpoint giai đoạn SSL (chưa có đầu phân "
                        "lớp) — bỏ qua.", d)
            return None
        from src.models.gat_transformer import GATTransformerIDS
        model = GATTransformerIDS(
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            num_classes=mc["num_classes"], backbone=mc.get("backbone", "negat"),
            hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
            edge_latent=mc.get("edge_latent", 24),
            node_heads=mc.get("node_heads", 2), heads=mc.get("heads", 3),
            num_layers=mc.get("num_layers", 3), dropout=mc.get("dropout", 0.3),
            edge_score_mode=mc.get("edge_score_mode", "full"),
            edge_agg=mc.get("edge_agg", "mean_deg"),
            edge_update=mc.get("edge_update", True),
            use_edge_features=mc.get("use_edge_features", True),
            d_model=mc.get("d_model", 32), nhead=mc.get("nhead", 4),
            tr_layers=mc.get("tr_layers", 2), dim_ff=mc.get("dim_ff", 64),
            mask_ratio=mc.get("mask_ratio", 0.3),
            clf_hidden=mc.get("clf_hidden", 128), verbose=False)
    else:
        from src.models.gnn_baselines import MODEL_REGISTRY
        key = {"mlp": "mlp", "deepmlp": "mlp", "gcn": "gcn",
               "gcnbaseline": "gcn", "e-graphsage": "edge_graphsage",
               "egraphsage": "edge_graphsage",
               "edge_graphsage": "edge_graphsage"}.get(mt)
        if key is None:
            log.warning("  Không nhận diện model type: %s (%s)", mt, d)
            return None
        model = MODEL_REGISTRY[key](
            in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
            hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
            num_classes=mc["num_classes"], dropout=mc["dropout"])

    sd = torch.load(w_p, map_location=device, weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        log.warning("  %s: trọng số lệch (thiếu %d, thừa %d)",
                    name, len(miss), len(unexp))
    return model.to(device).eval(), mc


# ════════════════════ trích biểu diễn và dự đoán ════════════════════
def _gnn_forward(model, graphs, device, mc, per_layer, bs=64):
    """Trả (dict tên_tầng -> Z, y_true, y_pred).

    per_layer=True: lấy biểu diễn SAU TỪNG TẦNG nếu mô hình hỗ trợ
    (return_all_layers / hidden list); nếu không, chỉ lấy tầng cuối.
    """
    from torch_geometric.loader import DataLoader
    is_gt = str(mc.get("model_type", "")).lower().startswith("gat-transformer")
    win = mc.get("window", 512)

    feats, Y, P = {}, [], []
    for batch in DataLoader(graphs, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        layer_z = None
        if is_gt:
            logits = model(batch.x, batch.edge_index, batch.edge_attr,
                           mode="clf", window=win)
            z = model.embed(batch.x, batch.edge_index, batch.edge_attr)
            layer_z = {"cuối": z}
        else:
            got = None
            if per_layer:
                # EdgeAwareGAT và NEGAT đều trả dict {h1..hK, final, logits}
                # khi gọi với return_embeddings=True.
                try:
                    got = model(batch.x, batch.edge_index, batch.edge_attr,
                                return_embeddings=True)
                except TypeError:
                    got = None
            if isinstance(got, dict) and "logits" in got:
                logits = got["logits"]
                layer_z = {}
                for i in range(1, 20):
                    k = "h%d" % i
                    if k in got:
                        layer_z["tầng %d" % i] = got[k]
                    else:
                        break
                layer_z["trước bộ phân loại"] = got["final"]
            else:
                out = model(batch.x, batch.edge_index, batch.edge_attr,
                            return_feat=True)
                if isinstance(out, tuple):
                    logits, z = out[0], out[1]
                else:
                    logits, z = out["logits"], out["final"]
                layer_z = {"tầng cuối": z}
        for k, v in layer_z.items():
            feats.setdefault(k, []).append(v.detach().cpu().numpy())
        Y.append(batch.y.cpu().numpy())
        P.append(logits.argmax(1).cpu().numpy())
    return ({k: np.concatenate(v) for k, v in feats.items()},
            np.concatenate(Y), np.concatenate(P))


def _tree_forward(loaded, graphs):
    """Cây: véc-tơ xác suất lớp đóng vai 'biểu diễn cuối'.

    SỬA LỖI v3.41: bản cũ dùng trực tiếp `g.x` (d_n cột), nhưng train_all.py
    huấn luyện cây trên `_flatten_graphs` — nối [x, trung bình láng giềng,
    cực đại láng giềng] nên số cột là 3·d_n. Với RPLGuard: 13 so với 39, và
    sklearn báo "X has 13 features, but RandomForestClassifier is expecting
    39 features". Phải dùng ĐÚNG hàm dựng đặc trưng của lúc huấn luyện.
    """
    from src.models.tabular_baselines import _flatten_graphs
    clf, inv_remap = loaded if isinstance(loaded, tuple) else (loaded, None)
    X, y = _flatten_graphs(graphs)
    pred_r = clf.predict(X)
    # Nhãn gốc: XGBoost/LightGBM dự đoán trong không gian đã đánh số lại.
    if inv_remap:
        pred = np.array([inv_remap.get(int(c), 0) for c in pred_r])
    else:
        pred = np.asarray(pred_r)
    try:
        Z = clf.predict_proba(X)
    except Exception:
        Z = np.eye(int(max(pred.max(), y.max())) + 1)[pred.astype(int)]
    return {"xác suất lớp": Z}, y, pred.astype(int)


# ════════════════════════════ vẽ ════════════════════════════
def _subsample(y, n_total, seed=42, min_per_class=25):
    rng = np.random.default_rng(seed)
    idx = []
    cls, cnt = np.unique(y, return_counts=True)
    budget = max(n_total // max(len(cls), 1), min_per_class)
    for c in cls:
        w = np.where(y == c)[0]
        k = min(len(w), budget)
        idx.append(rng.choice(w, k, replace=False))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def _tsne(X, perp, seed=42):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.nan_to_num(X))
    perp = float(min(perp, max(5, (len(X) - 1) / 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed, max_iter=1000).fit_transform(X)


def _silhouette(X2, y):
    from sklearn.metrics import silhouette_score
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X2, y))
    except Exception:
        return float("nan")


def plot_tsne(panels, y, names, out_png, suptitle, note=None):
    """panels: list[(tiêu_đề, X2)]."""
    n = len(panels)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.2 * rows),
                             squeeze=False)
    cls = np.unique(y)
    cmap = plt.get_cmap("tab20" if len(cls) > 10 else "tab10")
    for k, (title, X2) in enumerate(panels):
        ax = axes[k // cols][k % cols]
        for j, c in enumerate(cls):
            m = y == c
            lab = names[int(c)] if int(c) < len(names) else str(c)
            ax.scatter(X2[m, 0], X2[m, 1], s=5, alpha=0.65,
                       color=cmap(j % cmap.N), label=lab, linewidths=0)
        s = _silhouette(X2, y)
        ax.set_title(f"{title}\nsilhouette = {s:.3f}" if s == s else title,
                     fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_alpha(0.3)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(cls), 8), fontsize=8,
               markerscale=2.6, frameon=False,
               bbox_to_anchor=(0.5, -0.02 if rows == 1 else 0.0))
    fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=1.0)
    if note:
        fig.text(0.5, -0.06 if rows == 1 else -0.03, note, ha="center",
                 fontsize=8.5, style="italic", color="#b03a2e")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cm(y_true, y_pred, names, out_png, title, top=14):
    nc = len(names)
    cm = cm_fn(y_true, y_pred, nc)
    sup = cm.sum(1)
    keep = np.argsort(-sup)[:min(top, nc)]
    keep = np.sort(keep)
    sub = cm[np.ix_(keep, keep)]
    norm = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    lab = [names[i] for i in keep]

    fig, ax = plt.subplots(figsize=(max(6.5, 0.62 * len(keep) + 3.2),
                                    max(5.6, 0.56 * len(keep) + 2.6)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(keep))); ax.set_yticks(range(len(keep)))
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(lab, fontsize=8)
    for i in range(len(keep)):
        for j in range(len(keep)):
            v = norm[i, j]
            if v > 0.004:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "#222")
    acc = float((y_true == y_pred).mean())
    ax.set_xlabel("Dự đoán", fontsize=10)
    ax.set_ylabel("Nhãn thật", fontsize=10)
    ax.set_title(f"{title}\nAccuracy = {acc:.4f}  ·  {len(y_true):,} nút",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="tỉ lệ theo hàng")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return cm, acc


# ════════════════════════════ main ════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="t-SNE + ma trận nhầm lẫn cho MỌI mô hình trên MỘT bộ")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default=None,
                    help="thư mục checkpoint GAT-Transformer sau tinh chỉnh")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="giới hạn danh sách mô hình (mặc định: tất cả). "
                         "Nhận CẢ dấu cách và dấu phẩy: "
                         "--models RF XGBoost  hoặc  --models RF,XGBoost")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--skip-cm", action="store_true")
    ap.add_argument("--tag", default=None, help="nhãn bộ dữ liệu ghi trên hình")
    a = ap.parse_args()

    set_seed(a.seed)
    device = get_device("cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    names = meta["class_names"]
    graphs = torch.load(os.path.join(a.data_dir, f"{a.split}.pt"),
                        weights_only=False)
    tag = a.tag or os.path.basename(a.data_dir.rstrip("/"))
    log.info("Bộ %s | %s = %d đồ thị | %d lớp", tag, a.split, len(graphs),
             len(names))

    # SỬA LỖI v3.41: `nargs="*"` chỉ tách theo DẤU CÁCH, nên
    # `--models RF,XGBoost,LightGBM` trước đây thành MỘT tên mô hình duy
    # nhất "RF,XGBoost,LightGBM", không khớp gì và bị bỏ qua im lặng. Trong
    # khi train_all.py --models lại tách theo DẤU PHẨY. Chấp nhận cả hai
    # dạng và báo lỗi rõ nếu tên không hợp lệ, thay vì bỏ qua âm thầm.
    if a.models:
        toks = [t.strip() for tok in a.models for t in str(tok).split(",")]
        alias = {m.lower().replace("-", "").replace("_", ""): m
                 for m in ALL_MODELS}
        wanted, bad = [], []
        for t in toks:
            if not t:
                continue
            k = t.lower().replace("-", "").replace("_", "")
            (wanted.append(alias[k]) if k in alias else bad.append(t))
        if bad:
            raise SystemExit("Không có mô hình %s. Hợp lệ: %s"
                             % (", ".join(bad), ", ".join(ALL_MODELS)))
    else:
        wanted = list(ALL_MODELS)
    summary, failed, skipped = [], [], []

    for name in wanted:
        log.info("\n" + "=" * 62)
        log.info(">>> %s", name)
        try:
            if name in TREE:
                clf = _load_tree(name, a.ckpt_dir)
                if clf is None:
                    log.warning("  bỏ qua (chưa huấn luyện hoặc thiếu tệp)")
                    skipped.append(name)
                    continue
                feats, y, pred = _tree_forward(clf, graphs)
                mc = {"model_type": name}
            else:
                ov = a.gt_dir if name == "GAT-Transformer" else None
                got = _load_gnn(name, a.ckpt_dir, device, override_dir=ov)
                if got is None:
                    log.warning("  bỏ qua (chưa huấn luyện hoặc thiếu tệp)")
                    skipped.append(name)
                    continue
                model, mc = got
                feats, y, pred = _gnn_forward(
                    model, graphs, device, mc, per_layer=(name in PER_LAYER))
        except Exception as e:                      # pragma: no cover
            # In CẢ vết gọi. Bản cũ chỉ in một dòng nên hai lỗi thật (dict
            # chưa bóc vỏ, lệch 13 so với 39 cột) nằm im suốt nhiều lần
            # chạy: người dùng chỉ thấy "bỏ qua" và tưởng là chưa huấn
            # luyện. Một lỗi bị che là một lỗi không bao giờ được sửa.
            import traceback
            log.error("  LỖI khi xử lý %s: %s: %s",
                      name, type(e).__name__, e)
            log.error("  %s", traceback.format_exc().strip().replace(
                "\n", "\n  "))
            failed.append(name)
            continue

        # ── CHỈ SỐ: accuracy MỘT MÌNH gây hiểu nhầm khi lớp mất cân bằng ──
        # Đo được trên RPL-Beh: RF có accuracy 0,8807 (hạng 3) nhưng macro-F1
        # chỉ 0,5286 (hạng 6) — xếp hạng theo accuracy ĐẢO so với macro-F1.
        # Vì vậy ghi cả macro-F1 và F1 TỪNG LỚP.
        from sklearn.metrics import f1_score
        acc = float((y == pred).mean())
        lab = list(range(len(names)))
        f_cls = f1_score(y, pred, average=None, labels=lab, zero_division=0)
        macro = float(f1_score(y, pred, average="macro", labels=lab,
                               zero_division=0))
        per_class = {names[i]: round(float(f_cls[i]), 4) for i in lab}
        summary.append((name, acc, len(feats), macro, per_class))

        # ── ma trận nhầm lẫn ──
        if not a.skip_cm:
            p = os.path.join(a.out_dir, f"cm_{tag}_{name}.png")
            cm, _ = plot_cm(y, pred, names, p, f"{name} — {tag}")
            np.savetxt(os.path.join(a.out_dir, f"cm_{tag}_{name}.csv"),
                       cm, fmt="%d", delimiter=",",
                       header=",".join(names), comments="")
            log.info("  ma trận nhầm lẫn -> %s", os.path.basename(p))

        # ── t-SNE ──
        if not a.skip_tsne:
            sel = _subsample(y, a.n_samples, a.seed)
            ys = y[sel]
            panels = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for k, Z in feats.items():
                    panels.append((k, _tsne(Z[sel], a.perplexity, a.seed)))
            note = None
            if name in TREE:
                note = ("Mô hình cây không có tầng ẩn — trục t-SNE dựng từ "
                        "VÉC-TƠ XÁC SUẤT LỚP, không phải không gian nhúng.")
            kind = "từng tầng" if name in PER_LAYER else "tầng cuối"
            p = os.path.join(a.out_dir, f"tsne_{tag}_{name}.png")
            plot_tsne(panels, ys, names, p,
                      f"t-SNE {kind} — {name} — {tag}", note)
            log.info("  t-SNE (%s, %d bảng) -> %s", kind, len(panels),
                     os.path.basename(p))

    log.info("\n" + "=" * 62)
    log.info("TÓM TẮT — bộ %s", tag)
    # bảng tóm tắt: XẾP THEO macro-F1, không theo accuracy
    summary.sort(key=lambda r: -r[3])
    _nm = list(names)
    log.info("  %-18s %9s %9s | %s | %s", "mô hình", "Acc", "macro-F1",
             " ".join("%-7s" % c[:7] for c in _nm), "t-SNE")
    for n, acc, npan, macro, pc in summary:
        log.info("  %-18s %9.4f %9.4f | %s | %d bảng", n, acc, macro,
                 " ".join("%-7.4f" % pc[c] for c in _nm), npan)
    if len(summary) > 1:
        _a = sorted(summary, key=lambda r: -r[1])[0][0]
        _f = summary[0][0]
        if _a != _f:
            log.warning("  Xếp hạng theo ACCURACY (%s dẫn đầu) KHÁC xếp hạng "
                        "theo MACRO-F1 (%s dẫn đầu). Với dữ liệu mất cân bằng "
                        "lớp, macro-F1 mới là chỉ số nên báo cáo.", _a, _f)
    # Đối chiếu YÊU CẦU với THỰC TẾ. Không có dòng này thì một mô hình
    # thiếu chỉ thể hiện bằng việc KHÔNG xuất hiện — rất dễ bỏ qua.
    log.info("  ── đối chiếu: yêu cầu %d | thành công %d | chưa huấn luyện "
             "%d | LỖI %d", len(wanted), len(summary), len(skipped),
             len(failed))
    if skipped:
        log.warning("  chưa huấn luyện: %s", ", ".join(skipped))
    if failed:
        log.error("  LỖI (phải sửa, không phải thiếu checkpoint): %s",
                  ", ".join(failed))
    json.dump({"dataset": tag, "split": a.split,
               "requested": list(wanted), "untrained": skipped,
               "failed": failed,
               "class_names": list(names),
               "models": [{"name": n, "accuracy": a_, "macro_f1": mf,
                           "per_class_f1": pc, "tsne_panels": p}
                          for n, a_, p, mf, pc in summary]},
              open(os.path.join(a.out_dir, f"viz_summary_{tag}.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Đã lưu tất cả vào %s", a.out_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
