#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH_bo_ro_ri.py — LOẠI ĐẶC TRƯNG RÒ RỈ NHÃN KHỎI DỮ LIỆU ĐÃ TIỀN XỬ LÝ
========================================================================

Vì sao cần tệp này
------------------
Bước kiểm toán trong `convert_data.py` chỉ CẢNH BÁO chứ không tự loại đặc
trưng. Thông điệp cảnh báo trỏ tới `PATCH_bo_ro_ri.py` — nhưng tệp đó chưa
từng tồn tại trong kho mã. Đây chính là tệp bị thiếu.

Trường hợp đã phát hiện (bộ IoT-RPL 2021):

    f12   nMI=0.757   F1_nhị_phân_1_cột=0.914   <== RÒ RỈ
    WARNING — Đặc trưng RÒ RỈ: ['f12'] -> PHẢI loại khỏi dữ liệu

`f12` là cột thứ 13 (đánh số từ 0) của ma trận đặc trưng node, tức
`send_share = n_gửi / (n_gửi + n_nhận)`.

VÌ SAO NÓ RÒ RỈ — và một lưu ý quan trọng
-----------------------------------------
`send_share` KHÔNG phải một đặc trưng vô nghĩa. Nó là tín hiệu hành vi hợp
lệ: nút blackhole thật sự nhận mà không chuyển tiếp, nên tỉ lệ gửi của nó
tụt xuống gần 0; nút flooding thật sự phát nhiều nên tỉ lệ tăng gần 1.

Vấn đề nằm ở BỘ DỮ LIỆU chứ không ở đặc trưng: trong IoT-RPL 2021, vị trí
nút tấn công CỐ ĐỊNH ở mọi tệp mô phỏng và hành vi tấn công được cài đặt
tuyệt đối sạch (không nhiễu, không xác suất). Hệ quả là `send_share` trở
thành hàm gần như tất định của nhãn — một cây quyết định độ sâu 2 chỉ dùng
riêng cột này đã đạt F1 = 0,914.

Do đó, việc loại `send_share` KHÔNG phải là "sửa đặc trưng xấu", mà là một
phép thử: mô hình còn phát hiện được tấn công không, khi đã lấy đi manh mối
mà bộ dữ liệu vô tình để lộ. Cần trình bày đúng như vậy trong luận văn —
đừng viết rằng `send_share` là đặc trưng lỗi.

Cách dùng
---------
    # xem trước, không ghi gì
    python PATCH_bo_ro_ri.py --data-dir data/processed --dry-run

    # loại cột theo chỉ số (mặc định cho iotrpl là 12)
    python PATCH_bo_ro_ri.py --data-dir data/processed --drop 12 \
                             --out-dir data/processed_clean

    # loại theo TÊN đặc trưng (an toàn hơn — tự tra chỉ số)
    python PATCH_bo_ro_ri.py --data-dir data/processed --drop-name send_share \
                             --out-dir data/processed_clean

    # tự lấy danh sách rò rỉ từ kiểm toán chạy lại trên tập train
    python PATCH_bo_ro_ri.py --data-dir data/processed --auto \
                             --out-dir data/processed_clean

Sau đó chạy lại Bước 2 và Bước 3 trỏ vào thư mục mới:

    python scripts/train_all.py --data-dir data/processed_clean --epochs 120 --models all
    python scripts/test_all.py  --data-dir data/processed_clean

Tệp này KHÔNG ghi đè dữ liệu gốc trừ khi bạn chỉ định --out-dir trùng
--data-dir. Giữ lại bản gốc để còn tái lập được kết quả "trước khi vá".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Danh sách rò rỉ đã biết theo từng bộ dữ liệu (chỉ số cột đặc trưng NODE)
KNOWN_LEAKY = {
    "iotrpl": [12],          # send_share — F1 một cột 0,914
}


def _load(path):
    return torch.load(path, weights_only=False)


def _feature_names(dataset, d_n):
    """Trả tên cột nếu biết, để log cho người đọc hiểu đang bỏ cái gì."""
    try:
        if dataset == "iotrpl":
            from src.data.iotrpl_builder import NODE_FEATURE_NAMES as N
            if len(N) == d_n:
                return list(N)
        if dataset == "radar":
            from src.data.feature_audit import (RADAR_NODE_FEATURES_CLEAN,
                                                RADAR_NODE_FEATURES_V2)
            for N in (RADAR_NODE_FEATURES_V2, RADAR_NODE_FEATURES_CLEAN):
                if len(N) == d_n:
                    return list(N)
    except Exception:
        pass
    return [f"f{i}" for i in range(d_n)]


def drop_columns(graphs, drop_idx):
    """Loại các cột chỉ định khỏi `g.x` của mọi đồ thị. Trả về danh sách mới."""
    if not drop_idx:
        return graphs
    d = graphs[0].x.shape[1]
    keep = [i for i in range(d) if i not in set(drop_idx)]
    keep_t = torch.tensor(keep, dtype=torch.long)
    for g in graphs:
        g.x = g.x.index_select(1, keep_t).contiguous()
    return graphs


def audit_leaky(graphs, names, thr=0.90):
    """Chạy lại phép thử gốc-cây một cột, trả chỉ số các cột vượt ngưỡng."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score

    X = np.concatenate([g.x.numpy() for g in graphs])
    y = np.concatenate([g.y.numpy() for g in graphs])
    yb = (y > 0).astype(int)
    leaky = []
    for j in range(X.shape[1]):
        col = X[:, j:j + 1]
        dt = DecisionTreeClassifier(max_depth=2, random_state=0).fit(col, yb)
        f1 = f1_score(yb, dt.predict(col), zero_division=0)
        if f1 > thr:
            leaky.append((j, names[j], f1))
    return leaky


def audit_combo(graphs, names, depth=2, thr=0.95):
    """Phép thử RÒ RỈ TỔ HỢP: cây nông dùng TẤT CẢ cột.

    Vì sao cần: phép thử một cột chỉ bắt được đặc trưng nào TỰ MÌNH tái tạo
    nhãn. Nó KHÔNG bắt được trường hợp nhãn bị lộ qua một TỔ HỢP vài cột —
    tình huống điển hình khi bộ dữ liệu cố định danh tính nút tấn công: mỗi
    cột riêng lẻ chỉ mô tả hành vi, nhưng vài cột gộp lại thì chỉ đích danh
    được nút, và nút đó luôn thực hiện đúng một loại tấn công.

    Một cây độ sâu 2 chỉ tạo được tối đa 4 lá. Nếu 4 lá đã đủ tách gần hoàn
    hảo 5 lớp trên hàng nghìn mẫu thì bài toán không phải là dễ — nhãn đang
    bị lộ.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import f1_score

    X = np.concatenate([g.x.numpy() for g in graphs])
    y = np.concatenate([g.y.numpy() for g in graphs])
    out = {}
    for d in (1, 2, 3):
        dt = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        f1 = f1_score(y, dt.predict(X), average="macro", zero_division=0)
        out[d] = (f1, dt)
        flag = "  <== NGHI RÒ RỈ TỔ HỢP" if (d <= depth and f1 > thr) else ""
        print(f"  cây độ sâu {d}: macro-F1 = {f1:.4f}{flag}")
    _, best = out[depth]
    print(f"\n  Cây độ sâu {depth} dùng các cột:")
    used = sorted({names[i] for i in best.tree_.feature if i >= 0})
    print("   ", ", ".join(used))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Loại đặc trưng rò rỉ nhãn khỏi dữ liệu đã tiền xử lý")
    ap.add_argument("--data-dir", default="data/processed",
                    help="thư mục chứa train.pt / val.pt / test.pt / meta.json")
    ap.add_argument("--out-dir", default=None,
                    help="thư mục ghi kết quả (mặc định: <data-dir>_clean)")
    ap.add_argument("--drop", default="", help="chỉ số cột cần loại, vd: 12 hoặc 12,15")
    ap.add_argument("--drop-name", default="", help="tên cột cần loại, vd: send_share")
    ap.add_argument("--auto", action="store_true",
                    help="tự chạy kiểm toán trên tập train và loại mọi cột bị gắn cờ")
    ap.add_argument("--dataset", default=None,
                    help="ghi đè tên bộ dữ liệu (mặc định đọc từ meta.json)")
    ap.add_argument("--thr", type=float, default=0.90,
                    help="ngưỡng F1 một cột để coi là rò rỉ (mặc định 0,90)")
    ap.add_argument("--combo", action="store_true",
                    help="chạy thêm phép thử RÒ RỈ TỔ HỢP (cây nông dùng mọi cột)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in, không ghi tệp")
    a = ap.parse_args()

    meta_path = os.path.join(a.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"LỖI: không thấy {meta_path}. Chạy convert_data.py trước.")
        return 1
    meta = json.load(open(meta_path, encoding="utf-8"))
    dataset = a.dataset or meta.get("dataset", "?")

    train = _load(os.path.join(a.data_dir, "train.pt"))
    val = _load(os.path.join(a.data_dir, "val.pt"))
    test = _load(os.path.join(a.data_dir, "test.pt"))
    d_n = train[0].x.shape[1]
    names = _feature_names(dataset, d_n)

    print(f"Bộ dữ liệu : {dataset}")
    print(f"Đồ thị     : train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Đặc trưng  : d_n={d_n}  d_e={train[0].edge_attr.shape[1]}")

    # ── xác định danh sách cột cần loại ──
    drop = set()
    if a.drop:
        drop |= {int(s) for s in a.drop.replace(" ", "").split(",") if s}
    if a.drop_name:
        for nm in a.drop_name.replace(" ", "").split(","):
            if nm in names:
                drop.add(names.index(nm))
            else:
                print(f"  CẢNH BÁO: không thấy cột tên '{nm}' — bỏ qua")
    if a.auto:
        print("\nChạy lại kiểm toán trên tập TRAIN...")
        found = audit_leaky(train, names, a.thr)
        for j, nm, f1 in found:
            print(f"  RÒ RỈ: cột {j:2d}  {nm:<22s} F1_một_cột={f1:.3f}")
        drop |= {j for j, _, _ in found}
        if not found:
            print("  Không cột nào vượt ngưỡng.")
    if a.combo:
        print("\n── PHÉP THỬ RÒ RỈ TỔ HỢP (trên tập TRAIN) ──")
        audit_combo(train, names)
        print("  Nếu cây độ sâu 2 đã đạt macro-F1 > 0,95 thì việc loại thêm cột")
        print("  KHÔNG cứu được bộ dữ liệu: nhãn bị lộ qua tổ hợp, thường do")
        print("  danh tính nút tấn công cố định. Xem ghi chú cuối tệp này.")

    if not drop and dataset in KNOWN_LEAKY:
        drop = set(KNOWN_LEAKY[dataset])
        print(f"\nDùng danh sách rò rỉ đã biết cho '{dataset}': {sorted(drop)}")

    if not drop:
        print("\nKhông có cột nào để loại. Kết thúc.")
        return 0

    print("\n── SẼ LOẠI CÁC CỘT SAU ──")
    for j in sorted(drop):
        print(f"  cột {j:2d}  ->  {names[j]}")
    print(f"Số chiều đặc trưng node: {d_n} -> {d_n - len(drop)}")

    if a.dry_run:
        print("\n--dry-run: không ghi tệp nào.")
        return 0

    out = a.out_dir or (a.data_dir.rstrip("/\\") + "_clean")
    os.makedirs(out, exist_ok=True)

    for nm, gs in (("train", train), ("val", val), ("test", test)):
        drop_columns(gs, drop)
        torch.save(gs, os.path.join(out, nm + ".pt"))

    # cập nhật meta
    meta["n_node_features"] = d_n - len(drop)
    meta["dropped_leaky_idx"] = sorted(drop)
    meta["dropped_leaky_names"] = [names[j] for j in sorted(drop)]
    meta["node_feature_names"] = [n for i, n in enumerate(names) if i not in drop]
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # scaler: chép nguyên vẹn NHƯNG cắt các tham số theo cột đã loại
    sp = os.path.join(a.data_dir, "scaler.pt")
    if os.path.exists(sp):
        sc = torch.load(sp, weights_only=False)
        keep = [i for i in range(d_n) if i not in drop]
        for k in ("x_lam", "x_mean", "x_std"):
            if k in sc and sc[k] is not None and len(np.asarray(sc[k])) == d_n:
                sc[k] = np.asarray(sc[k])[keep]
        torch.save(sc, os.path.join(out, "scaler.pt"))
        print("  scaler.pt: đã cắt tham số theo cột còn lại")

    print(f"\n═══ ĐÃ GHI -> {out} ═══")
    print("Bước tiếp:")
    print(f"  python scripts/train_all.py --data-dir {out} --epochs 120 --models all")
    print(f"  python scripts/test_all.py  --data-dir {out}")
    print("\nLƯU Ý khi viết luận văn: nếu sau khi loại cột mà độ chính xác vẫn")
    print("tuyệt đối 1,0000 ở nhiều họ mô hình, rò rỉ CHƯA hết — chạy lại với")
    print("--auto để tìm cột tiếp theo, và xem lại thiết kế bộ dữ liệu.")
    return 0


__doc_khi_van_ro_ri__ = """
KHI LOẠI CỘT MÀ VẪN CÒN RÒ RỈ
=============================
Trường hợp đã gặp: sau khi loại `send_share` khỏi bộ IoT-RPL 2021, độ chính
xác vẫn tuyệt đối 1,0000 ở ba họ mô hình khác nhau (RF, E-GraphSAGE,
EdgeGAT). Nghĩa là rò rỉ KHÔNG nằm ở một cột đơn lẻ.

Nguyên nhân gốc nằm ở THIẾT KẾ bộ dữ liệu, không ở đặc trưng: trong
IoT-RPL 2021, mỗi nút tấn công giữ NGUYÊN một vai trò ở MỌI tệp mô phỏng
(một nút luôn là blackhole, một nút khác luôn là rank...). Bất kỳ tổ hợp
đặc trưng nào đủ để nhận ra "đây là nút số 12" cũng đồng thời tiết lộ nhãn.
Không có phép loại cột nào chữa được điều này, vì thông tin danh tính nằm
rải rác trong nhiều cột hành vi hợp lệ.

Ba lựa chọn, theo thứ tự ưu tiên:

1. LOẠI BỘ DỮ LIỆU KHỎI PHẦN KẾT LUẬN. Đây là lựa chọn trung thực và cũng
   là lựa chọn mạnh nhất về mặt học thuật: trình bày chính quá trình phát
   hiện như một đóng góp phương pháp luận, và dùng RADAR + UOS làm bằng
   chứng. Cả hai bộ này đều có vị trí kẻ tấn công thay đổi theo tệp.

2. Nếu vẫn muốn giữ: chỉ dùng nó như bộ dữ liệu MINH HOẠ, báo cáo kèm cảnh
   báo rõ ràng, và KHÔNG dùng để so sánh xếp hạng giữa các mô hình.

3. Chia tập theo NÚT TẤN CÔNG thay vì theo tệp — nhưng với bộ này thì bất
   khả thi, vì mỗi loại tấn công chỉ gắn với đúng một nút, nên giữ nút đó
   ra khỏi tập huấn luyện đồng nghĩa với việc lớp đó biến mất khỏi train.
   Chính sự bất khả thi này là bằng chứng cho kết luận ở mục 1.
"""


if __name__ == "__main__":
    sys.exit(main())
