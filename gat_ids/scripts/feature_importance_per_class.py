#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_importance_per_class.py — ĐẶC TRƯNG NÀO PHÂN BIỆT TỪNG LOẠI TẤN CÔNG
============================================================================

Vì sao cần tệp riêng
--------------------
`feature_audit.py` trả lời câu hỏi "đặc trưng nào là bản sao của nhãn" (rò rỉ).
Nó KHÔNG trả lời câu hỏi "đặc trưng nào giúp phân biệt tấn công A với phần còn
lại". Hai câu hỏi khác nhau: một cái tìm thứ phải LOẠI BỎ, cái kia tìm thứ mô
hình thực sự DỰA VÀO.

Phương pháp
-----------
Với mỗi lớp tấn công c, dựng bài toán MỘT-ĐẤU-PHẦN-CÒN-LẠI (one-vs-rest) và đo
tầm quan trọng của từng đặc trưng theo BA phương pháp độc lập, rồi lấy hạng
trung bình. Dùng ba phương pháp vì mỗi phương pháp có thiên lệch riêng:

  1. Độ giảm bất thuần Gini của Random Forest — nhanh, nhưng thiên vị các
     đặc trưng có nhiều giá trị khác nhau (high-cardinality).
  2. Hoán vị (permutation importance) trên TẬP KIỂM THỬ — đo mức tụt hiệu năng
     thật khi xáo trộn một cột; không thiên vị cardinality, nhưng chia đều
     công trạng khi hai cột tương quan mạnh.
  3. Chênh lệch trung bình chuẩn hoá (Cohen's d) giữa lớp c và phần còn lại —
     hoàn toàn không dùng mô hình, nên bắt được cả quan hệ mà cây bỏ sót.

Chỉ báo cáo đặc trưng lọt top-k ở ÍT NHẤT HAI trong ba phương pháp. Một đặc
trưng chỉ nổi ở một phương pháp thường là hiện tượng riêng của phương pháp đó.

Cách dùng
---------
    python scripts/feature_importance_per_class.py --data-dir data/processed
    python scripts/feature_importance_per_class.py --data-dir data/processed \\
           --edge --top 8 --out bang_dactrung_radar.csv

Sinh ra: bảng in ra màn hình + tệp CSV để dán thẳng vào luận văn.

Ghi chú diễn giải — ĐỌC TRƯỚC KHI VIẾT VÀO LUẬN VĂN
---------------------------------------------------
Kết quả của tệp này cho biết đặc trưng nào mang THÔNG TIN phân biệt một lớp,
theo góc nhìn của mô hình cây. Nó KHÔNG phải bằng chứng rằng EdgeGAT dùng đúng
những đặc trưng đó: EdgeGAT học biểu diễn phi tuyến qua ba tầng truyền thông
điệp, nên đặc trưng nó dựa vào có thể khác. Muốn nói về chính EdgeGAT, phải
dùng trọng số chú ý hoặc GNNExplainer (scripts/analyze_model.py).

Cách viết đúng: "đặc trưng X mang nhiều thông tin phân biệt lớp Y nhất trong
bộ dữ liệu này", KHÔNG viết "mô hình dùng đặc trưng X để phát hiện Y".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_names(dataset, d_n, d_e, want_edge):
    """Lấy tên đặc trưng thật; nếu không khớp chiều thì trả f0..fN."""
    try:
        if dataset == "radar":
            from src.data.feature_audit import (RADAR_NODE_FEATURES_V29,
                                                RADAR_NODE_FEATURES_CLEAN21)
            from src.data.radar_builder import RADAR_EDGE_FEATURE_NAMES as RE
            if want_edge and len(RE) == d_e:
                return list(RE)
            for N in (RADAR_NODE_FEATURES_V29, RADAR_NODE_FEATURES_CLEAN21):
                if len(N) == d_n:
                    return list(N)
        if dataset == "iotrpl":
            from src.data.iotrpl_builder import (NODE_FEATURE_NAMES as NN,
                                                 EDGE_FEATURE_NAMES as NE)
            if want_edge and len(NE) == d_e:
                return list(NE)
            if len(NN) == d_n:
                return list(NN)
        if dataset == "uos":
            from src.data.uos_builder import (UOS_NODE_FEATURE_NAMES as NN,
                                              UOS_EDGE_FEATURE_NAMES as NE)
            if want_edge and len(NE) == d_e:
                return list(NE)
            if len(NN) == d_n:
                return list(NN)
    except Exception:
        pass
    d = d_e if want_edge else d_n
    return [f"f{i}" for i in range(d)]


def flatten_nodes(graphs):
    X = np.concatenate([g.x.numpy() for g in graphs])
    y = np.concatenate([g.y.numpy() for g in graphs])
    return X, y


def flatten_edges(graphs):
    """Gán nhãn cho CẠNH = nhãn của node đích (đầu nhận thông điệp).

    Quy ước này có lý do: trong bài toán phân loại node, một cạnh chỉ ảnh
    hưởng tới nhãn qua việc nó đóng góp thông điệp cho node đích. Cần nêu rõ
    quy ước khi báo cáo, vì một cạnh nối nút thường với nút tấn công sẽ được
    đếm hai lần với hai nhãn khác nhau (mỗi chiều một lần).
    """
    Xs, ys = [], []
    for g in graphs:
        ea = g.edge_attr.numpy()
        dst = g.edge_index[1].numpy()
        lab = g.y.numpy()
        m = dst < len(lab)
        Xs.append(ea[m]); ys.append(lab[dst[m]])
    return np.concatenate(Xs), np.concatenate(ys)


def cohens_d(X, mask):
    """|Cohen's d| giữa nhóm mask và phần còn lại, cho từng cột."""
    a, b = X[mask], X[~mask]
    if len(a) < 2 or len(b) < 2:
        return np.zeros(X.shape[1])
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    sp = np.where(sp < 1e-12, np.nan, sp)
    d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / sp
    return np.nan_to_num(d)


def analyse_class(Xtr, ytr, Xte, yte, cls, names, top, n_est, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score

    btr, bte = (ytr == cls), (yte == cls)
    if btr.sum() < 20 or bte.sum() < 5:
        return None

    rf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced",
                                n_jobs=-1, random_state=seed,
                                min_samples_leaf=2).fit(Xtr, btr)
    f1 = f1_score(bte, rf.predict(Xte), zero_division=0)

    gini = rf.feature_importances_
    perm = permutation_importance(rf, Xte, bte, n_repeats=5,
                                  random_state=seed, n_jobs=-1,
                                  scoring="f1").importances_mean
    coh = cohens_d(Xtr, btr)

    # hạng: 0 = quan trọng nhất
    rk = lambda v: np.argsort(np.argsort(-v))
    r_g, r_p, r_c = rk(gini), rk(perm), rk(coh)
    votes = (r_g < top).astype(int) + (r_p < top).astype(int) + (r_c < top).astype(int)
    mean_rank = (r_g + r_p + r_c) / 3.0

    order = np.argsort(mean_rank)
    rows = []
    for j in order:
        if votes[j] >= 2:
            rows.append(dict(feature=names[j], idx=int(j),
                             gini=float(gini[j]), perm=float(perm[j]),
                             cohen_d=float(coh[j]), votes=int(votes[j]),
                             mean_rank=float(mean_rank[j])))
        if len(rows) >= top:
            break
    return dict(f1_ovr=float(f1), n_train=int(btr.sum()), n_test=int(bte.sum()),
                rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--edge", action="store_true",
                    help="phân tích đặc trưng CẠNH thay vì đặc trưng NODE")
    ap.add_argument("--top", type=int, default=6, help="số đặc trưng báo cáo mỗi lớp")
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--max-rows", type=int, default=250000,
                    help="lấy mẫu ngẫu nhiên nếu dữ liệu quá lớn (0 = dùng hết)")
    ap.add_argument("--out", default=None, help="ghi bảng ra tệp CSV")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    cls_names = meta["class_names"]
    dataset = meta.get("dataset", "?")

    train = torch.load(os.path.join(a.data_dir, "train.pt"), weights_only=False)
    test = torch.load(os.path.join(a.data_dir, "test.pt"), weights_only=False)

    if a.edge:
        Xtr, ytr = flatten_edges(train); Xte, yte = flatten_edges(test)
    else:
        Xtr, ytr = flatten_nodes(train); Xte, yte = flatten_nodes(test)

    if a.max_rows and len(Xtr) > a.max_rows:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(Xtr), a.max_rows, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
        print(f"(lấy mẫu {a.max_rows} hàng huấn luyện cho nhanh)")

    names = load_names(dataset, Xtr.shape[1] if not a.edge else 0,
                       Xtr.shape[1] if a.edge else 0, a.edge)
    if len(names) != Xtr.shape[1]:
        names = [f"f{i}" for i in range(Xtr.shape[1])]

    kind = "CẠNH" if a.edge else "NODE"
    print(f"\nBộ dữ liệu: {dataset} | đặc trưng {kind}: {Xtr.shape[1]} cột")
    print(f"train {Xtr.shape} · test {Xte.shape}")
    if a.edge:
        print("Quy ước: nhãn của một cạnh = nhãn của NODE ĐÍCH.")

    out_rows = []
    for c, cname in enumerate(cls_names):
        if c == 0:
            continue                     # bỏ lớp Normal
        res = analyse_class(Xtr, ytr, Xte, yte, c, names, a.top, a.n_estimators)
        print("\n" + "═" * 74)
        if res is None:
            print(f"{cname}: quá ít mẫu, bỏ qua")
            continue
        print(f"{cname}  —  F1 một-đấu-phần-còn-lại = {res['f1_ovr']:.3f}  "
              f"(train {res['n_train']} / test {res['n_test']} mẫu)")
        print(f"  {'Đặc trưng':<24}{'Gini':>9}{'Hoán vị':>10}{'|d|':>9}{'Phiếu':>7}")
        print("  " + "-" * 60)
        for r in res["rows"]:
            print(f"  {r['feature']:<24}{r['gini']:>9.4f}{r['perm']:>10.4f}"
                  f"{r['cohen_d']:>9.3f}{r['votes']:>7d}/3")
            out_rows.append(dict(lop=cname, f1_ovr=round(res["f1_ovr"], 4), **r))
        if not res["rows"]:
            print("  (không đặc trưng nào lọt top ở ≥2 phương pháp — "
                  "lớp này phân biệt bằng TỔ HỢP nhiều đặc trưng yếu)")

    print("\n" + "═" * 74)
    print("CÁCH ĐỌC")
    print("  |d| < 0,2 : chênh lệch không đáng kể   |  0,2–0,5 : nhỏ")
    print("  0,5–0,8   : trung bình                 |  > 0,8   : lớn")
    print("  Hoán vị = mức TỤT F1 khi xáo trộn cột đó trên tập kiểm thử;")
    print("  giá trị âm nghĩa là cột đó không đóng góp (nhiễu).")
    print("  F1 một-đấu-phần-còn-lại thấp ⟹ lớp đó khó tách bằng đặc trưng đơn lẻ,")
    print("  bảng bên dưới khi đó ít ý nghĩa dù các con số vẫn hiện.")
    print("\nLƯU Ý: bảng này nói về THÔNG TIN có trong dữ liệu, không phải về")
    print("cách EdgeGAT thực sự quyết định. Muốn nói về EdgeGAT, dùng trọng số")
    print("chú ý hoặc GNNExplainer (scripts/analyze_model.py).")

    if a.out and out_rows:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"\nĐã ghi -> {a.out}")


if __name__ == "__main__":
    main()
