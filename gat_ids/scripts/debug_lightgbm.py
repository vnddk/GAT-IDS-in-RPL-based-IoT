#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_lightgbm.py — KHOANH VÙNG NGUYÊN NHÂN LIGHTGBM SỤP TRÊN BỘ RADAR
=====================================================================

Bối cảnh
--------
Trên bộ RADAR (16 lớp), LightGBM cho accuracy 0,1061 / macro-F1 0,0754 —
thấp hơn cả một bộ phân loại luôn đoán lớp đa số. Trong khi cùng mã đó cho
0,8713 trên UOS và 1,0000 trên IoT-RPL.

ĐÃ LOẠI TRỪ (kiểm chứng bằng dữ liệu tổng hợp, tỉ lệ mất cân bằng tới 694:1):
  - `class_weight='balanced'` KHÔNG gây sụp: acc 0,986 ở tỉ lệ 694:1.
  - Truyền thừa `objective='multiclass'` và `num_class=...` KHÔNG đổi kết quả.
Nghĩa là nguyên nhân nằm ở thứ gì đó riêng của RADAR, không tái lập được
nếu không có chính bộ dữ liệu đó. Tệp này chạy TRÊN MÁY BẠN để tìm ra.

Cách dùng
---------
    python scripts/debug_lightgbm.py --data-dir data/processed

Script chạy 6 phép thử theo thứ tự và in kết luận. Câu hỏi then chốt là
phép thử 1: **accuracy trên chính TẬP HUẤN LUYỆN**.

  - Nếu acc(train) CAO mà acc(test) thấp  -> lỗi ở khâu dự đoán / dữ liệu
    test / vòng ánh xạ nhãn, KHÔNG phải ở khâu học.
  - Nếu acc(train) cũng THẤP             -> mô hình không học được; vấn đề
    ở siêu tham số hoặc ở chính dữ liệu huấn luyện.

Đây là thông tin mà nhật ký hiện tại không hề in ra — và cũng là lý do lỗi
này chỉ lộ ra ở Bước 3 thay vì bị bắt ngay ở Bước 2.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.tabular_baselines import _flatten_graphs   # noqa: E402


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(y, p, C):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=list(range(C)), zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--sub", type=int, default=0,
                    help="chỉ dùng N đồ thị đầu để chạy nhanh (0 = dùng hết)")
    a = ap.parse_args()

    from lightgbm import LGBMClassifier
    import lightgbm
    print("Phiên bản LightGBM:", lightgbm.__version__)

    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    C = meta["num_classes"]
    train = torch.load(os.path.join(a.data_dir, "train.pt"), weights_only=False)
    test = torch.load(os.path.join(a.data_dir, "test.pt"), weights_only=False)
    if a.sub:
        train, test = train[:a.sub], test[:a.sub]

    X_tr, y_tr = _flatten_graphs(train)
    X_te, y_te = _flatten_graphs(test)
    print(f"train {X_tr.shape}  test {X_te.shape}  num_classes={C}")
    print("phân bố lớp train:", dict(zip(*np.unique(y_tr, return_counts=True))))
    print("phân bố lớp test :", dict(zip(*np.unique(y_te, return_counts=True))))

    # ── PHÉP THỬ 0: dữ liệu có bất thường không ──
    print("\n[0] Kiểm tra dữ liệu")
    for nm, X in (("train", X_tr), ("test", X_te)):
        print(f"  {nm}: NaN={np.isnan(X).sum()}  Inf={np.isinf(X).sum()}  "
              f"min={np.nanmin(X):.3g}  max={np.nanmax(X):.3g}")
    tr_classes = set(np.unique(y_tr).tolist())
    te_classes = set(np.unique(y_te).tolist())
    if te_classes - tr_classes:
        print(f"  !! Lớp CÓ trong test nhưng KHÔNG có trong train: "
              f"{sorted(te_classes - tr_classes)}")
        print("     -> remap sẽ thiếu mục, `inv_remap.get(c, 0)` âm thầm gán về lớp 0.")

    base = dict(n_estimators=300, num_leaves=63, max_depth=-1, learning_rate=0.08,
                subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)

    present = sorted(tr_classes)
    remap = {c: i for i, c in enumerate(present)}
    inv = {i: c for c, i in remap.items()}
    y_tr_r = np.array([remap[c] for c in y_tr])

    # ── PHÉP THỬ 1: acc trên chính TẬP HUẤN LUYỆN (câu hỏi then chốt) ──
    print("\n[1] Cấu hình HIỆN TẠI — đo trên train VÀ test")
    t0 = time.time()
    m1 = LGBMClassifier(class_weight="balanced", objective="multiclass",
                        num_class=len(present), **base).fit(X_tr, y_tr_r)
    p_tr = np.array([inv[c] for c in m1.predict(X_tr)])
    p_te = np.array([inv.get(c, 0) for c in m1.predict(X_te)])
    a_tr, a_te = acc(y_tr, p_tr), acc(y_te, p_te)
    print(f"  acc(TRAIN)={a_tr:.4f}   acc(TEST)={a_te:.4f}   "
          f"macroF1(TEST)={macro_f1(y_te, p_te, C):.4f}   ({time.time()-t0:.0f}s)")
    if a_tr > 0.7 and a_te < 0.3:
        print("  => KẾT LUẬN: mô hình HỌC ĐƯỢC nhưng SỤP trên test.")
        print("     Nguyên nhân nằm ở phía dữ liệu test hoặc khâu dự đoán,")
        print("     KHÔNG phải ở siêu tham số. Xem tiếp [4] và [5].")
    elif a_tr < 0.3:
        print("  => KẾT LUẬN: mô hình KHÔNG học được ngay trên tập huấn luyện.")
        print("     Vấn đề ở khâu học. Xem tiếp [2] và [3].")

    # ── PHÉP THỬ 2: bỏ class_weight ──
    print("\n[2] Bỏ class_weight='balanced'")
    m2 = LGBMClassifier(**base).fit(X_tr, y_tr_r)
    p2 = np.array([inv.get(c, 0) for c in m2.predict(X_te)])
    print(f"  acc(TEST)={acc(y_te,p2):.4f}  macroF1={macro_f1(y_te,p2,C):.4f}")

    # ── PHÉP THỬ 3: bỏ objective/num_class truyền tay ──
    print("\n[3] Bỏ objective/num_class truyền tay (giữ class_weight)")
    m3 = LGBMClassifier(class_weight="balanced", **base).fit(X_tr, y_tr_r)
    p3 = np.array([inv.get(c, 0) for c in m3.predict(X_te)])
    print(f"  acc(TEST)={acc(y_te,p3):.4f}  macroF1={macro_f1(y_te,p3,C):.4f}")

    # ── PHÉP THỬ 4: KHÔNG remap nhãn ──
    print("\n[4] Huấn luyện trực tiếp trên nhãn gốc (không remap)")
    m4 = LGBMClassifier(class_weight="balanced", **base).fit(X_tr, y_tr)
    p4 = m4.predict(X_te)
    print(f"  acc(TEST)={acc(y_te,p4):.4f}  macroF1={macro_f1(y_te,p4,C):.4f}")
    if acc(y_te, p4) - acc(y_te, p_te) > 0.3:
        print("  => Vòng remap/inv_remap là thủ phạm. Bỏ nó đi.")

    # ── PHÉP THỬ 5: pickle vòng tròn ──
    print("\n[5] Kiểm tra pickle vòng tròn (lưu rồi nạp lại)")
    blob = pickle.dumps({"model": m3, "remap": remap})
    m5 = pickle.loads(blob)["model"]
    p5 = np.array([inv.get(c, 0) for c in m5.predict(X_te)])
    same = int((p5 == p3).all())
    print(f"  acc(TEST) sau pickle={acc(y_te,p5):.4f}  "
          f"trùng khớp dự đoán trước pickle: {'CÓ' if same else 'KHÔNG'}")
    if not same:
        print("  => Pickle làm hỏng mô hình. Dùng clf.booster_.save_model() thay vì pickle.")

    # ── PHÉP THỬ 6: đối chứng bằng RF trên đúng cùng ma trận ──
    print("\n[6] Đối chứng — RandomForest trên ĐÚNG cùng X_tr/X_te")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    prf = rf.predict(X_te)
    print(f"  acc(TEST)={acc(y_te,prf):.4f}  macroF1={macro_f1(y_te,prf,C):.4f}")
    print("  (nếu RF bình thường mà LightGBM sụp -> lỗi riêng của LightGBM;")
    print("   nếu CẢ HAI cùng sụp -> vấn đề nằm ở dữ liệu, không ở mô hình)")

    # ── TỔNG KẾT ──
    print("\n" + "=" * 62)
    print("TỔNG KẾT — chọn cấu hình có acc(TEST) cao nhất trong các phép thử trên")
    print("và áp lại vào khối LightGBM ở scripts/train_all.py.")
    print("Nhớ chạy lại CẢ Bước 2 lẫn Bước 3 sau khi sửa.")


if __name__ == "__main__":
    main()
