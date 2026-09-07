#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsne_compare.py — So sánh KHÔNG GIAN ĐẶC TRƯNG CUỐI giữa nhiều mô hình
======================================================================

Trích biểu diễn ẩn ngay TRƯỚC bộ phân loại của từng mô hình trên cùng một
tập kiểm thử, rồi vẽ t-SNE cạnh nhau kèm điểm silhouette.

Dùng để trả lời câu hỏi: hai kiến trúc khác nhau có tạo ra không gian đặc
trưng khác nhau về chất hay không, và ATA có làm cụm chặt hơn không.

HAI QUY TẮC PHƯƠNG PHÁP LUẬN được cưỡng chế trong mã:

1. Điểm silhouette LUÔN tính trên KHÔNG GIAN GỐC (32 chiều), không bao giờ
   tính trên toạ độ t-SNE hai chiều. Lý do: t-SNE chỉ bảo toàn quan hệ láng
   giềng cục bộ, không bảo toàn khoảng cách toàn cục, nên mọi số đo lấy từ
   toạ độ t-SNE đều không có giá trị định lượng. Hình t-SNE ở đây chỉ để
   QUAN SÁT ĐỊNH TÍNH.

2. Mọi mô hình dùng CHUNG một tập mẫu con và CHUNG một hạt giống t-SNE, nên
   khác biệt nhìn thấy giữa các bảng là do MÔ HÌNH chứ không do phép lấy mẫu.

Cách dùng
---------
    python scripts/tsne_compare.py --data-dir data/radar_chrono \\
        --models "EdgeGAT=ck_edge" "EdgeGAT+ATA=ck_ata" \\
                 "NE-GAT=ck_negat" "NE-GAT+ATA=ck_negat_ata" \\
        --n-samples 4000 --out tsne_compare.png
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils.common import get_device, get_logger, set_seed   # noqa: E402

log = get_logger()
plt.rcParams["font.family"] = "DejaVu Sans"


def load_any(ckpt_dir, device):
    """Nạp mô hình từ thư mục checkpoint, tự nhận diện kiến trúc."""
    mc_path = os.path.join(ckpt_dir, "model_config.json")
    if not os.path.exists(mc_path):
        for cand in ("EdgeGAT", "NE-GAT", "."):
            p = os.path.join(ckpt_dir, cand, "model_config.json")
            if os.path.exists(p):
                mc_path = p
                break
    mc = json.load(open(mc_path, encoding="utf-8"))
    w = os.path.join(os.path.dirname(mc_path), "best_model.pt")
    mt = str(mc.get("model_type", "EdgeGAT")).lower()

    if mt in ("ne-gat", "negat", "ne_gat"):
        from src.models.ne_gat import NEGAT as Cls
    else:
        from src.models.edge_gat import EdgeAwareGAT as Cls
    ok = set(inspect.signature(Cls.__init__).parameters) - {"self"}
    kw = {k: v for k, v in mc.items() if k in ok}
    if "verbose" in ok:
        kw["verbose"] = False
    model = Cls(**kw)
    model.load_state_dict(torch.load(w, map_location=device))
    return model.to(device).eval(), mc


@torch.no_grad()
def extract(model, graphs, device, bs=64):
    """Trả (Z, y) với Z là biểu diễn ngay trước bộ phân loại."""
    Z, Y = [], []
    for batch in DataLoader(graphs, batch_size=bs, shuffle=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, return_feat=True)
        z = out[1] if isinstance(out, tuple) else out["final"]
        Z.append(z.detach().cpu().numpy())
        Y.append(batch.y.cpu().numpy())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models", nargs="+", required=True,
                    help='dạng "Nhãn=thư_mục_checkpoint", cách nhau bởi dấu cách')
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tsne_compare.png")
    ap.add_argument("--csv", default=None, help="ghi bảng silhouette ra CSV")
    a = ap.parse_args()

    set_seed(a.seed)
    device = get_device("auto")
    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    names = meta["class_names"]
    test = torch.load(os.path.join(a.data_dir, "test.pt"), weights_only=False)

    specs = []
    for m in a.models:
        if "=" not in m:
            raise SystemExit(f'--models cần dạng "Nhãn=thư_mục", nhận được: {m}')
        lab, d = m.split("=", 1)
        specs.append((lab, d))

    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # ── trích đặc trưng của mọi mô hình trên CÙNG tập nút ──
    feats, labels_ref, sel = {}, None, None
    for lab, d in specs:
        model, mc = load_any(d, device)
        Z, y = extract(model, test, device)
        if labels_ref is None:
            labels_ref = y
            rng = np.random.default_rng(a.seed)
            n = min(a.n_samples, len(y))
            # lấy mẫu PHÂN TẦNG để lớp hiếm không biến mất khỏi hình
            idx = []
            for c in np.unique(y):
                ci = np.where(y == c)[0]
                take = max(1, int(round(n * len(ci) / len(y))))
                idx += list(rng.choice(ci, size=min(take, len(ci)), replace=False))
            sel = np.array(sorted(idx))
            log.info("Lấy mẫu phân tầng %d nút trong %d nút kiểm thử", len(sel), len(y))
        elif not np.array_equal(y, labels_ref):
            raise SystemExit("Các mô hình cho thứ tự nhãn khác nhau — "
                             "kiểm tra lại cùng data-dir.")
        feats[lab] = Z[sel]
        log.info("%-22s d=%d  tham số=%s", lab, Z.shape[1],
                 f"{sum(p.numel() for p in model.parameters()):,}")

    ys = labels_ref[sel]

    # ── silhouette trên KHÔNG GIAN GỐC ──
    sil = {}
    for lab, Z in feats.items():
        sil[lab] = float(silhouette_score(Z, ys)) if len(np.unique(ys)) > 1 else float("nan")

    # ── t-SNE, cùng hạt giống cho mọi mô hình ──
    n = len(specs)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 6.0 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("tab20")
    for k, (lab, _d) in enumerate(specs):
        ax = axes[k // ncol][k % ncol]
        emb = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
                   random_state=a.seed, max_iter=1000).fit_transform(feats[lab])
        for c in np.unique(ys):
            m = ys == c
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=.75,
                       color=("#9AA3AD" if c == 0 else cmap((c - 1) % 20)),
                       label=names[c] if c < len(names) else str(c),
                       linewidths=0)
        ax.set_title(f"{lab}\nsilhouette (không gian gốc) = {sil[lab]:+.4f}",
                     fontsize=12.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_color("#DDDDDD")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(8, len(l)), fontsize=9,
               frameon=False, markerscale=2.4, bbox_to_anchor=(.5, .035))
    fig.suptitle("So sánh không gian đặc trưng cuối — t-SNE (tô màu theo lớp thật)",
                 fontsize=14.5, fontweight="bold", y=.995)
    fig.text(.5, .006,
             "Hình t-SNE chỉ để QUAN SÁT ĐỊNH TÍNH. Mọi kết luận định lượng lấy từ "
             "điểm silhouette, vốn được tính trên KHÔNG GIAN GỐC.",
             ha="center", fontsize=9.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0, .10, 1, .97])
    fig.savefig(a.out, dpi=200, facecolor="white")
    log.info("Đã lưu hình -> %s", a.out)

    print("\n" + "=" * 64)
    print("SILHOUETTE TRÊN KHÔNG GIAN GỐC (càng cao càng tách cụm rõ)")
    print("=" * 64)
    for lab, v in sorted(sil.items(), key=lambda t: -t[1]):
        bar = "▇" * max(0, int((v + 0.2) * 40))
        print(f"  {lab:<24}{v:+.4f}  {bar}")
    print("\nLƯU Ý: silhouette đo CẤU TRÚC HÌNH HỌC của biểu diễn, độc lập với")
    print("bộ phân loại. Silhouette cao không tự động kéo theo macro-F1 cao;")
    print("hai độ đo bổ sung cho nhau, cần đọc cùng nhau.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["model", "silhouette_original_space"])
            for lab, v in sil.items():
                w.writerow([lab, round(v, 6)])
        log.info("Đã ghi bảng -> %s", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
