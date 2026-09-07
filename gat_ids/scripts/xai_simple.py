#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_simple.py — BIỂU ĐỒ ĐƠN GIẢN: MÔ HÌNH DỰA VÀO GÌ, SO VỚI DỮ LIỆU CÓ GÌ
===========================================================================

Tệp này thay thế bản in bảng số dày đặc (hệ số d, Gini, hoán vị, Spearman)
bằng MỘT thứ duy nhất: biểu đồ cột nằm ngang, cho mỗi lớp tấn công, đặt cạnh
nhau hai nguồn bằng chứng.

    ┌─────────────────────────┬─────────────────────────┐
    │  MÔ HÌNH  (EdgeGAT)     │  DỮ LIỆU  (Random Forest)│
    │  top 10 đặc trưng NÚT   │  top 10 đặc trưng NÚT    │
    │  top 6 đặc trưng CẠNH   │  top 6 đặc trưng CẠNH    │
    └─────────────────────────┴─────────────────────────┘

Đặc trưng xuất hiện ở CẢ HAI cột được tô ĐẬM và ghi "TRÙNG"; đặc trưng chỉ
có ở một bên được ghi "LỆCH". Đó là toàn bộ điều cần đọc.

  · TRÙNG ⟹ mô hình khai thác đúng tín hiệu mạnh nhất có trong dữ liệu.
  · LỆCH  ⟹ đáng viết: mô hình tìm được tín hiệu khác, hoặc bỏ lỡ tín hiệu
            mà dữ liệu có sẵn.

Điểm số được chuẩn hoá về thang 0–1 trong từng cột (chia cho giá trị lớn
nhất), nên chỉ so sánh được THỨ HẠNG giữa hai bên, không so sánh được giá trị
tuyệt đối. Đây là chủ ý: hai bên đo bằng đơn vị khác nhau nên so giá trị
tuyệt đối là vô nghĩa.

Cách dùng
---------
  # bước 1 — lấy phía DỮ LIỆU (Random Forest)
  python scripts/feature_importance_per_class.py --data-dir data/processed \\
         --top 10 --out dactrung_node.csv
  python scripts/feature_importance_per_class.py --data-dir data/processed \\
         --edge --top 6 --out dactrung_canh.csv

  # bước 2 — lấy phía MÔ HÌNH (GNNExplainer + chú ý)
  python scripts/attention_per_class.py --data-dir data/processed \\
         --ckpt-dir ck_mlp --top 10 --export-model-csv model_xai.csv

  # bước 3 — vẽ biểu đồ so sánh
  python scripts/xai_simple.py --model-csv model_xai.csv \\
         --rf-node-csv dactrung_node.csv --rf-edge-csv dactrung_canh.csv \\
         --out-dir hinh_xai
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"

C_MODEL, C_RF, C_MATCH = "#1C7293", "#DD8452", "#065A82"


def norm(v):
    v = np.asarray(v, dtype=float)
    m = np.abs(v).max()
    return np.abs(v) / m if m > 0 else v * 0


def panel(ax, feats, vals, title, color, matched, xlabel):
    y = np.arange(len(feats))[::-1]
    cols = [C_MATCH if f in matched else color for f in feats]
    ax.barh(y, vals, color=cols, edgecolor="#2A2A2A", lw=0.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [(f + "  ✓" if f in matched else f) for f in feats],
        fontsize=8.5,
        fontweight=None)
    for yy, v, f in zip(y, vals, feats):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=7.2,
                color="#333333")
    ax.set_xlim(0, 1.16)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_class(cls, mn, mv, rn, rv, men, mev, ren, rev, out_dir):
    """mn/mv: node mô hình · rn/rv: node RF · men/mev: cạnh mô hình · ren/rev: cạnh RF"""
    match_n = set(mn) & set(rn)
    match_e = set(men) & set(ren)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2),
                             gridspec_kw={"height_ratios": [10, 6.4]})
    fig.suptitle(f"Lớp tấn công: {cls}", fontsize=14, fontweight="bold", y=0.985)

    panel(axes[0][0], mn, mv, "MÔ HÌNH EdgeGAT — top đặc trưng NÚT\n(GNNExplainer)",
          C_MODEL, match_n, "mức đóng góp (chuẩn hoá)")
    panel(axes[0][1], rn, rv, "DỮ LIỆU — top đặc trưng NÚT\n(Random Forest một-đấu-phần-còn-lại)",
          C_RF, match_n, "mức đóng góp (chuẩn hoá)")
    if len(men):
        panel(axes[1][0], men, mev, "MÔ HÌNH EdgeGAT — top đặc trưng CẠNH\n(ảnh hưởng tới hệ số chú ý)",
              C_MODEL, match_e, "mức đóng góp (chuẩn hoá)")
    else:
        axes[1][0].axis("off")
        axes[1][0].text(0.5, 0.5, "(không có dữ liệu cạnh cho lớp này)",
                        ha="center", fontsize=9, color="#888888")
    if len(ren):
        panel(axes[1][1], ren, rev, "DỮ LIỆU — top đặc trưng CẠNH\n(Random Forest một-đấu-phần-còn-lại)",
              C_RF, match_e, "mức đóng góp (chuẩn hoá)")
    else:
        axes[1][1].axis("off")

    n_all = len(set(mn) | set(rn))
    txt = (f"NÚT: {len(match_n)}/{len(mn)} đặc trưng TRÙNG giữa hai bên"
           f"   ·   CẠNH: {len(match_e)}/{max(len(men),1)} TRÙNG"
           "      (dấu ✓ và cột màu đậm = trùng)")
    fig.text(0.5, 0.015, txt, ha="center", fontsize=9.5, color="#065A82",
             fontweight="bold")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    p = os.path.join(out_dir, f"xai_{cls}.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p, len(match_n), len(mn), len(match_e), len(men)


def draw_summary(rows, out_dir):
    """Một hình tổng hợp: tỉ lệ trùng khớp của từng lớp."""
    rows = sorted(rows, key=lambda r: -(r[1] / max(r[2], 1)))
    cls = [r[0] for r in rows]
    pct = [100.0 * r[1] / max(r[2], 1) for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.42 * len(cls) + 1.6)))
    y = np.arange(len(cls))[::-1]
    cols = [C_MATCH if p >= 50 else (C_RF if p >= 20 else "#B23A3A") for p in pct]
    ax.barh(y, pct, color=cols, edgecolor="#2A2A2A", lw=0.5, height=0.7)
    for yy, p, r in zip(y, pct, rows):
        ax.text(p + 1.2, yy, f"{r[1]}/{r[2]}", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(cls, fontsize=9.5)
    ax.set_xlim(0, 108); ax.set_xlabel("Tỉ lệ đặc trưng NÚT trùng nhau giữa mô hình và dữ liệu (%)",
                                       fontsize=9.5)
    ax.set_title("Mức đồng thuận giữa điều MÔ HÌNH dùng và điều DỮ LIỆU có",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.015,
             "Cao = mô hình khai thác đúng tín hiệu mạnh nhất trong dữ liệu   ·   "
             "Thấp = mô hình dùng tín hiệu khác (đáng phân tích)",
             ha="center", fontsize=9, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    p = os.path.join(out_dir, "xai_tong_hop.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-csv", required=True,
                    help="CSV từ attention_per_class.py --export-model-csv")
    ap.add_argument("--rf-node-csv", required=True)
    ap.add_argument("--rf-edge-csv", default=None)
    ap.add_argument("--out-dir", default="hinh_xai")
    ap.add_argument("--top-node", type=int, default=10)
    ap.add_argument("--top-edge", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    M = pd.read_csv(a.model_csv)
    RN = pd.read_csv(a.rf_node_csv)
    RE = pd.read_csv(a.rf_edge_csv) if a.rf_edge_csv and os.path.exists(a.rf_edge_csv) else None

    # phía RF: dùng cột mean_rank (nhỏ = quan trọng) -> đổi thành điểm
    def rf_top(df, cls, k):
        d = df[df["lop"] == cls]
        if d.empty:
            return [], []
        d = d.sort_values("mean_rank").head(k)
        sc = norm(1.0 / (d["mean_rank"].to_numpy() + 1.0))
        return list(d["feature"]), list(sc)

    def md_top(df, cls, kind, k):
        d = df[(df["lop"] == cls) & (df["kind"] == kind)]
        if d.empty:
            return [], []
        d = d.sort_values("score", ascending=False).head(k)
        return list(d["feature"]), list(norm(d["score"].to_numpy()))

    rows, made = [], []
    for cls in sorted(M["lop"].unique()):
        mn, mv = md_top(M, cls, "node", a.top_node)
        rn, rv = rf_top(RN, cls, a.top_node)
        men, mev = md_top(M, cls, "edge", a.top_edge)
        ren, rev = rf_top(RE, cls, a.top_edge) if RE is not None else ([], [])
        if not mn or not rn:
            print(f"  bỏ qua {cls}: thiếu dữ liệu một phía")
            continue
        p, k1, n1, k2, n2 = draw_class(cls, mn, mv, rn, rv, men, mev, ren, rev, a.out_dir)
        rows.append((cls, k1, n1))
        made.append(p)
        print(f"  {cls:<22} NÚT trùng {k1}/{n1}   CẠNH trùng {k2}/{max(n2,1)}   -> {os.path.basename(p)}")

    if rows:
        sp = draw_summary(rows, a.out_dir)
        print(f"\nHình tổng hợp -> {sp}")
    print(f"\nĐã tạo {len(made)} hình trong {a.out_dir}/")
    print("\nCÁCH ĐỌC:")
    print("  Cột màu đậm + dấu ✓  = đặc trưng có mặt ở CẢ hai bên (TRÙNG).")
    print("  Chỉ so THỨ HẠNG giữa hai cột, KHÔNG so giá trị tuyệt đối —")
    print("  hai bên đo bằng đơn vị khác nhau.")


if __name__ == "__main__":
    main()
