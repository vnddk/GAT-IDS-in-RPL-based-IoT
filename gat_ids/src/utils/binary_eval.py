# -*- coding: utf-8 -*-
"""
binary_eval.py — Đánh giá NHỊ PHÂN (tấn công / bình thường)
============================================================

Bài toán được huấn luyện ở chế độ đa lớp, nhưng trong vận hành thực tế câu
hỏi đầu tiên luôn là *"nút này có đang tấn công hay không"*. Hai chế độ đánh
giá trả lời hai câu hỏi khác nhau và **không thay thế được nhau**:

    Đa lớp   ->  "đây là loại tấn công nào?"      (macro-F1 trên C lớp)
    Nhị phân ->  "có tấn công hay không?"          (F1 trên lớp dương)

Quy ước gộp: mọi lớp khác 0 được coi là lớp DƯƠNG.

    ŷ_bin = 1[ŷ > 0],      y_bin = 1[y > 0]

MỘT LƯU Ý QUAN TRỌNG VỀ CÁCH ĐỌC. Chỉ số nhị phân LUÔN cao hơn macro-F1 đa
lớp, vì gộp lớp xoá bỏ toàn bộ lỗi nhầm GIỮA các loại tấn công. Một mô hình
nhầm `rank` thành `sinkhole` bị phạt ở chế độ đa lớp nhưng ĐƯỢC TÍNH ĐÚNG ở
chế độ nhị phân. Do đó không được dùng con số nhị phân để tuyên bố mô hình
tốt hơn thực tế; phải báo cáo cả hai cạnh nhau.
"""
from __future__ import annotations

import numpy as np


def binary_metrics(y_true, y_pred, positive_from=1):
    """Trả về bộ chỉ số nhị phân, gộp mọi lớp >= positive_from thành DƯƠNG."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    yt = (y_true >= positive_from).astype(int)
    yp = (y_pred >= positive_from).astype(int)

    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(yt), 1)
    # tỉ lệ báo động sai: trong số nút BÌNH THƯỜNG, bao nhiêu bị báo nhầm
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    # tỉ lệ bỏ lọt: trong số nút TẤN CÔNG, bao nhiêu bị bỏ qua
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    # macro-F1 nhị phân: trung bình F1 của cả hai lớp, phạt cả hai chiều
    p0 = tn / (tn + fn) if (tn + fn) else 0.0
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) else 0.0
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "macro_f1": (f1 + f1_0) / 2.0, "f1_normal": f1_0,
        "fpr": fpr, "fnr": fnr, "mcc": float(mcc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def format_binary(m, title="ĐÁNH GIÁ NHỊ PHÂN (tấn công / bình thường)"):
    L = []
    L.append("  " + title)
    L.append("  " + "-" * 58)
    L.append(f"  {'Accuracy':<22}{m['accuracy']:.4f}")
    L.append(f"  {'Macro-F1 (2 lớp)':<22}{m['macro_f1']:.4f}")
    L.append(f"  {'F1 lớp TẤN CÔNG':<22}{m['f1']:.4f}")
    L.append(f"  {'F1 lớp Bình thường':<22}{m['f1_normal']:.4f}")
    L.append(f"  {'Precision (tấn công)':<22}{m['precision']:.4f}")
    L.append(f"  {'Recall (tấn công)':<22}{m['recall']:.4f}")
    L.append(f"  {'MCC':<22}{m['mcc']:+.4f}")
    L.append("  " + "-" * 58)
    L.append(f"  {'Tỉ lệ báo động sai':<22}{m['fpr']:.4f}   "
             f"({m['fp']:,} / {m['fp']+m['tn']:,} nút bình thường)")
    L.append(f"  {'Tỉ lệ bỏ lọt':<22}{m['fnr']:.4f}   "
             f"({m['fn']:,} / {m['fn']+m['tp']:,} nút tấn công)")
    L.append("  " + "-" * 58)
    L.append(f"  Ma trận nhầm lẫn 2x2:  TN={m['tn']:,}  FP={m['fp']:,}  "
             f"FN={m['fn']:,}  TP={m['tp']:,}")
    L.append("  Lưu ý: chỉ số nhị phân LUÔN cao hơn macro-F1 đa lớp vì phép gộp")
    L.append("  xoá bỏ lỗi nhầm GIỮA các loại tấn công. Phải đọc cùng nhau.")
    return "\n".join(L)
