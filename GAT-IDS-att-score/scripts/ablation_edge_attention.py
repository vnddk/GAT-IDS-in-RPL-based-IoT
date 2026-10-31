#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_edge_attention.py — VÌ SAO ĐƯA ĐẶC TRƯNG CẠNH VÀO HỆ SỐ CHÚ Ý LÀ ĐÚNG
==============================================================================

Câu hỏi cần trả lời chặt chẽ
----------------------------
Bảng so sánh bảy mô hình ở Chương 4 cho thấy EdgeGAT > E-GraphSAGE > GCN. Nhưng
ba mô hình đó khác nhau ở NHIỀU THỨ cùng lúc — kiểu tầng tích chập, cách tổng
hợp, số tham số — nên không thể quy chênh lệch về riêng một nguyên nhân.

Tệp này chạy một THIẾT KẾ GIAI THỪA 2×2 trên ĐÚNG MỘT bộ khung: cùng lớp
chiếu đầu vào, cùng DyT, cùng phần dư, cùng bộ phân loại, cùng hàm mất mát,
cùng số epoch, cùng hạt giống. Chỉ hai yếu tố thay đổi:

                        │ KHÔNG chú ý (SAGEConv,   │ CÓ chú ý (GATv2,
                        │ tổng hợp trung bình)     │ trọng số học được)
    ────────────────────┼──────────────────────────┼────────────────────────
    KHÔNG đặc trưng cạnh│  N  (đối chứng nền)      │  A
    CÓ đặc trưng cạnh   │  E                       │  AE  (mô hình đề xuất)

Bốn nhánh cho phép tách ba đại lượng:

    Hiệu ứng chính của CẠNH   = ((E − N) + (AE − A)) / 2
    Hiệu ứng chính của CHÚ Ý  = ((A − N) + (AE − E)) / 2
    TƯƠNG TÁC                 =  (AE − A) − (E − N)

Tương tác là con số quan trọng nhất. Giả thuyết của luận văn là nó DƯƠNG:
đặc trưng cạnh có ích hơn khi cơ chế chú ý hiện diện, vì lúc đó mô hình dùng
được chúng để QUYẾT ĐỊNH trọng số láng giềng, chứ không chỉ mang thêm thông
tin vào thông điệp. Nếu tương tác xấp xỉ 0, hai yếu tố chỉ cộng dồn và luận
điểm "đưa cạnh vào hệ số chú ý" mất phần lớn sức nặng — khi đó phải viết lại
Chương 5 cho trung thực.

Điểm mấu chốt về công bằng: ở hai nhánh KHÔNG chú ý (N và E), SAGEConv không
nhận edge_attr. Để nhánh E vẫn thực sự "có đặc trưng cạnh", script GỘP đặc
trưng cạnh vào đặc trưng NÚT trước khi huấn luyện (trung bình các cạnh tới và
đi của mỗi nút). Nhờ đó E có ĐÚNG lượng thông tin cạnh như AE, chỉ khác ở chỗ
thông tin đó không được dùng để tính trọng số tổng hợp. Đây chính là điều cần
cô lập.

Cách dùng
---------
    python scripts/ablation_edge_attention.py --data-dir data/processed \\
           --epochs 120 --seeds 1

    # khuyến nghị cho bản nộp: nhiều hạt giống
    python scripts/ablation_edge_attention.py --data-dir data/processed \\
           --epochs 120 --seeds 5 --out ablation_2x2.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils.common import load_config, set_seed, get_device, get_logger   # noqa: E402
from src.data.loader import make_loaders                                      # noqa: E402
from src.models.edge_gat import build_model                                   # noqa: E402
from src.utils.engine import make_criterion, train_centralized                # noqa: E402

log = get_logger()

ARMS = [
    ("N",  False, "sage",  "không cạnh · không chú ý"),
    ("E",  True,  "sage",  "CÓ cạnh · không chú ý"),
    ("A",  False, "gatv2", "không cạnh · CÓ chú ý"),
    ("AE", True,  "gatv2", "CÓ cạnh · CÓ chú ý  (đề xuất)"),
]


def merge_edge_into_nodes(graphs):
    """Gộp đặc trưng cạnh vào đặc trưng nút (trung bình cạnh vào + cạnh ra).

    Dùng cho nhánh E: SAGEConv không nhận edge_attr, nên nếu không gộp thì
    nhánh E chẳng khác gì nhánh N và phép so sánh trở nên vô nghĩa.
    Mỗi nút được nối thêm 2·d_e cột: trung bình cạnh ĐI và trung bình cạnh ĐẾN.
    """
    out = []
    for g in graphs:
        g2 = copy.copy(g)
        n, de = g.x.shape[0], g.edge_attr.shape[1]
        src, dst = g.edge_index[0], g.edge_index[1]
        agg_out = torch.zeros(n, de)
        agg_in = torch.zeros(n, de)
        cnt_out = torch.zeros(n, 1)
        cnt_in = torch.zeros(n, 1)
        agg_out.index_add_(0, src, g.edge_attr)
        agg_in.index_add_(0, dst, g.edge_attr)
        cnt_out.index_add_(0, src, torch.ones(len(src), 1))
        cnt_in.index_add_(0, dst, torch.ones(len(dst), 1))
        agg_out = agg_out / cnt_out.clamp(min=1)
        agg_in = agg_in / cnt_in.clamp(min=1)
        g2.x = torch.cat([g.x, agg_out, agg_in], dim=1)
        out.append(g2)
    return out


def run_arm(arm, use_edge, conv, cfg, data, meta, device, seed):
    train_g, val_g, test_g = data
    m2 = copy.deepcopy(meta)

    if conv == "sage" and use_edge:
        train_g = merge_edge_into_nodes(train_g)
        val_g = merge_edge_into_nodes(val_g)
        test_g = merge_edge_into_nodes(test_g)
        m2["n_node_features"] = train_g[0].x.shape[1]

    c2 = copy.deepcopy(cfg)
    c2["model"]["use_edge_features"] = bool(use_edge and conv == "gatv2")
    c2["model"]["conv_type"] = conv
    c2["train"]["ckpt_dir"] = os.path.join(cfg["train"]["ckpt_dir"],
                                           f"abl_{arm}_s{seed}")
    os.makedirs(c2["train"]["ckpt_dir"], exist_ok=True)

    set_seed(seed)
    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"])
    model = build_model(c2, m2).to(device)
    crit = make_criterion(c2, train_g, m2["num_classes"], device)
    model, test_m = train_centralized(model, loaders, crit, c2, m2, device)
    n_par = sum(p.numel() for p in model.parameters())
    return dict(macro_f1=float(test_m["macro_f1"]),
                accuracy=float(test_m["accuracy"]),
                params=int(n_par),
                per_class=test_m.get("per_class_f1", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--ckpt-dir", default="checkpoints/ablation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["train"]["epochs"] = a.epochs
    cfg["train"]["ckpt_dir"] = a.ckpt_dir
    device = get_device(cfg["device"])

    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    load = lambda n: torch.load(os.path.join(a.data_dir, n), weights_only=False)
    data = (load("train.pt"), load("val.pt"), load("test.pt"))
    log.info("Ablation 2×2 | train=%d val=%d test=%d | d_n=%d d_e=%d | %d hạt giống",
             len(data[0]), len(data[1]), len(data[2]),
             meta["n_node_features"], meta["n_edge_features"], a.seeds)

    res = {k: [] for k, _, _, _ in ARMS}
    detail = []
    for seed in range(42, 42 + a.seeds):
        for arm, ue, conv, desc in ARMS:
            log.info("\n═══ Nhánh %s (%s) — hạt giống %d ═══", arm, desc, seed)
            r = run_arm(arm, ue, conv, cfg, data, meta, device, seed)
            res[arm].append(r["macro_f1"])
            detail.append(dict(seed=seed, arm=arm, desc=desc, **{
                k: v for k, v in r.items() if k != "per_class"}))
            log.info("  -> macro-F1 = %.4f | acc = %.4f | %d tham số",
                     r["macro_f1"], r["accuracy"], r["params"])

    mean = {k: float(np.mean(v)) for k, v in res.items()}
    std = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in res.items()}

    print("\n" + "═" * 72)
    print("BẢNG GIAI THỪA 2×2 — macro-F1 trên tập kiểm thử")
    print("═" * 72)
    fmt = lambda k: (f"{mean[k]:.4f}" + (f" ± {std[k]:.4f}" if a.seeds > 1 else ""))
    print(f"{'':<22}{'KHÔNG chú ý':>20}{'CÓ chú ý':>20}")
    print(f"{'KHÔNG đặc trưng cạnh':<22}{fmt('N'):>20}{fmt('A'):>20}")
    print(f"{'CÓ đặc trưng cạnh':<22}{fmt('E'):>20}{fmt('AE'):>20}")

    eff_edge = ((mean["E"] - mean["N"]) + (mean["AE"] - mean["A"])) / 2
    eff_attn = ((mean["A"] - mean["N"]) + (mean["AE"] - mean["E"])) / 2
    inter = (mean["AE"] - mean["A"]) - (mean["E"] - mean["N"])

    print("\n" + "─" * 72)
    print(f"Hiệu ứng chính của ĐẶC TRƯNG CẠNH : {eff_edge:+.4f}")
    print(f"Hiệu ứng chính của CƠ CHẾ CHÚ Ý   : {eff_attn:+.4f}")
    print(f"TƯƠNG TÁC (cạnh × chú ý)          : {inter:+.4f}")
    print("─" * 72)
    print(f"  Cạnh giúp khi KHÔNG có chú ý : {mean['E'] - mean['N']:+.4f}   (E − N)")
    print(f"  Cạnh giúp khi CÓ chú ý       : {mean['AE'] - mean['A']:+.4f}   (AE − A)")

    if inter > 0.01:
        print("\n=> TƯƠNG TÁC DƯƠNG. Đặc trưng cạnh có ích HƠN khi có cơ chế chú ý.")
        print("   Đây là bằng chứng trực tiếp cho luận điểm của luận văn: giá trị")
        print("   nằm ở việc DÙNG đặc trưng cạnh để quyết định trọng số láng giềng,")
        print("   không chỉ ở việc mang thêm thông tin vào thông điệp.")
    elif inter < -0.01:
        print("\n=> TƯƠNG TÁC ÂM. Đặc trưng cạnh có ích HƠN khi KHÔNG có chú ý.")
        print("   Kết quả này NGƯỢC với giả thuyết. Phải báo cáo trung thực và")
        print("   viết lại phần lập luận ở Chương 5.")
    else:
        print("\n=> TƯƠNG TÁC ~ 0. Hai yếu tố chỉ CỘNG DỒN, không cộng hưởng.")
        print("   Luận điểm 'đưa cạnh vào hệ số chú ý' mất phần lớn sức nặng:")
        print("   lợi ích đến từ việc CÓ đặc trưng cạnh, không từ chỗ đưa chúng")
        print("   vào đâu. Cần viết lại Chương 5 cho đúng.")

    if a.seeds == 1:
        print("\nCẢNH BÁO: chỉ MỘT hạt giống. Các hiệu ứng trên chưa có ước lượng")
        print("độ biến thiên, nên chưa thể kết luận hiệu ứng nào là thật. Chạy lại")
        print("với --seeds 5 trước khi trích vào luận văn.")

    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
            w.writeheader(); w.writerows(detail)
        print(f"\nĐã ghi -> {a.out}")


if __name__ == "__main__":
    main()
