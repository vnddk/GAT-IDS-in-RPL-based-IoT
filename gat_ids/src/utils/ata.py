# -*- coding: utf-8 -*-
"""
ata.py — Adaptive Temporal Alignment (ATA) cho GAT-IDS
=======================================================

Cài đặt phương pháp ATA theo Phan, Dang Le, Vu và Vo (2026),
"Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift".

Ý TƯỞNG CỐT LÕI
---------------
Dưới phép đánh giá theo thứ tự thời gian, dạng thất bại chính KHÔNG phải là
dung lượng của bộ phân loại, mà là BIỂU DIỄN KHÔNG ỔN ĐỊNH giữa các giai đoạn
thời gian. ATA xử lý điều đó bằng cách coi tập huấn luyện là một tập hợp CÓ
THỨ TỰ gồm nhiều miền nguồn, thay vì một kho mẫu đồng nhất:

    L_ATA  =  Σ_k L_cls^(k)  +  (λ_d / |P|) · Σ_(a,b)∈P  L_trans^(a,b)

    · Số hạng thứ nhất giữ cho bộ mã hoá còn phân biệt được TRONG TỪNG giai
      đoạn, nên việc căn chỉnh không thể đạt được bằng cách làm sụp cấu trúc lớp.
    · Số hạng thứ hai lấy trung bình độ chênh lệch trên toàn bộ |P| = K(K−1)/2
      cặp miền, kéo phân bố đặc trưng của các giai đoạn về một hình học chung.
    · Hệ số 1/|P| giữ độ lớn của số hạng căn chỉnh không đổi khi K thay đổi.

Điểm phân biệt với DANN/CDAN/CORAL cổ điển: các phương pháp đó là
nguồn→đích và cần mẫu của miền ĐÍCH tại thời điểm huấn luyện. Trong bài toán
tiến-theo-thời-gian, miền đích là lưu lượng TƯƠNG LAI — chưa xảy ra nên không
thể lấy mẫu. ATA KHÔNG cần miền đích: nó ép tính nhất quán lẫn nhau giữa các
giai đoạn quá khứ đã quan sát được.

ĐIỀU CHỈNH CHO BÀI TOÁN ĐỒ THỊ
-------------------------------
Bài báo gốc làm việc trên chuỗi mẫu dạng bảng. Ở đây mỗi mẫu là một ĐỒ THỊ và
nhãn nằm ở TỪNG NÚT, nên có hai điều chỉnh:

1. Miền thời gian được dựng theo CHỈ SỐ CỬA SỔ TRONG TỪNG TỆP mô phỏng, không
   theo một đồng hồ toàn cục. Lý do vật lý: mỗi tệp RADAR là một lần chạy mô
   phỏng ĐỘC LẬP với đồng hồ riêng, nên không tồn tại thứ tự thời gian toàn
   cục giữa các tệp. Miền k gồm đoạn thứ k của MỌI tệp.

2. Đặc trưng đem đi căn chỉnh là biểu diễn ẩn ở MỨC NÚT sau tầng chú ý cuối,
   gộp trên toàn bộ nút của lô. Đây là biểu diễn mà bộ phân loại thực sự nhìn
   thấy, nên là chỗ đúng để áp ràng buộc ổn định.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import get_logger

log = get_logger()


# ══════════════════════════════════════════════════════════════════
#  Các độ đo chênh lệch giữa hai miền
# ══════════════════════════════════════════════════════════════════
def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """CORAL — khớp ma trận hiệp phương sai bậc hai.

        L = ‖ C_a − C_b ‖_F²  /  (4 d²)

    Rẻ nhất trong ba độ đo và không có siêu tham số. Đây là lựa chọn mặc định.
    """
    d = zs.size(1)
    if zs.size(0) < 2 or zt.size(0) < 2:
        return zs.new_zeros(())
    cs = torch.cov(zs.T)
    ct = torch.cov(zt.T)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def mmd_loss(zs: torch.Tensor, zt: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """MMD với nhân Gauss đa thang đo.

    Nhạy hơn CORAL vì bắt được chênh lệch ở mọi bậc mô-men, nhưng tốn
    O(n²) bộ nhớ nên cần giới hạn kích thước lô.
    """
    if zs.size(0) < 2 or zt.size(0) < 2:
        return zs.new_zeros(())

    def _k(a, b):
        d2 = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d2 / (2.0 * s * s)) for s in sigmas) / len(sigmas)

    return _k(zs, zs).mean() + _k(zt, zt).mean() - 2.0 * _k(zs, zt).mean()


def cosine_disc(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Chênh lệch cô-sin giữa hai véc-tơ trung tâm miền.

    Rẻ nhất và ổn định nhất với lô nhỏ, nhưng chỉ bắt được chênh lệch bậc một.
    """
    if zs.size(0) < 1 or zt.size(0) < 1:
        return zs.new_zeros(())
    ms, mt = zs.mean(0), zt.mean(0)
    return 1.0 - torch.nn.functional.cosine_similarity(ms, mt, dim=0).clamp(-1, 1)


DISCREPANCY = {"coral": coral_loss, "mmd": mmd_loss, "cosine": cosine_disc}


# ══════════════════════════════════════════════════════════════════
#  Dựng miền thời gian
# ══════════════════════════════════════════════════════════════════
def _graph_time_key(g, fallback_idx):
    """Khoá thời gian của một đồ thị: (tệp, chỉ số cửa sổ)."""
    f = getattr(g, "src_file", None)
    w = getattr(g, "win_idx", None)
    if w is None:
        w = getattr(g, "time_win", None)
    if w is None:
        w = fallback_idx          # thứ tự trong danh sách là phương án cuối
    return (f if f is not None else "_", int(w))


def assign_temporal_domains(graphs, K=3, strategy="quantile", disc="coral",
                            model=None, device=None):
    """Gán nhãn miền thời gian 0..K-1 cho từng đồ thị.

    strategy = "quantile": chia mỗi tệp thành K đoạn thời gian xấp xỉ bằng
        nhau theo chỉ số cửa sổ. Đây là chiến lược bài báo dùng cho kết quả
        chính, và là mặc định ở đây.

    strategy = "tdc": Temporal Distribution Characterization — chọn ranh giới
        sao cho TỔNG chênh lệch từng cặp giữa các đoạn là lớn nhất, tức làm
        nổi bật tính không dừng bên trong giai đoạn huấn luyện. Chỉ dùng đặc
        trưng và thứ tự thời gian của TẬP HUẤN LUYỆN; không dùng mẫu kiểm
        định hay kiểm thử.

    Trả về mảng numpy độ dài len(graphs) chứa chỉ số miền.
    """
    if K <= 1:
        return np.zeros(len(graphs), dtype=int)

    keys = [_graph_time_key(g, i) for i, g in enumerate(graphs)]
    files = {}
    for i, (f, w) in enumerate(keys):
        files.setdefault(f, []).append((w, i))

    dom = np.zeros(len(graphs), dtype=int)

    if strategy == "quantile":
        for f, items in files.items():
            items.sort()                      # theo chỉ số cửa sổ
            n = len(items)
            for pos, (_w, i) in enumerate(items):
                dom[i] = min(int(pos * K / n), K - 1)

    elif strategy == "tdc":
        # Ranh giới chọn theo chênh lệch đặc trưng, tính trên đặc trưng NÚT
        # trung bình của từng đồ thị (rẻ, không cần chạy mô hình).
        fn = DISCREPANCY[disc]
        for f, items in files.items():
            items.sort()
            n = len(items)
            if n < K * 2:
                for pos, (_w, i) in enumerate(items):
                    dom[i] = min(int(pos * K / n), K - 1)
                continue
            feats = torch.stack([graphs[i].x.mean(0) for _w, i in items])
            grid = np.linspace(0, n, 20 + 1)[1:-1].astype(int)
            grid = sorted(set(int(x) for x in grid if 0 < x < n))
            chosen = []
            for _ in range(K - 1):            # tham lam
                best, best_v = None, -1e18
                for c in grid:
                    if c in chosen:
                        continue
                    bs = sorted(chosen + [c])
                    segs, prev = [], 0
                    for b in bs + [n]:
                        if b - prev >= 2:
                            segs.append(feats[prev:b])
                        prev = b
                    if len(segs) < 2:
                        continue
                    v = sum(float(fn(segs[a], segs[b_]))
                            for a in range(len(segs))
                            for b_ in range(a + 1, len(segs)))
                    if v > best_v:
                        best_v, best = v, c
                if best is None:
                    break
                chosen.append(best)
            bs = sorted(chosen)
            for pos, (_w, i) in enumerate(items):
                k = sum(1 for b in bs if pos >= b)
                dom[i] = min(k, K - 1)
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy}")

    cnt = np.bincount(dom, minlength=K)
    log.info("ATA: %d miền thời gian (%s) trên %d tệp | số đồ thị mỗi miền: %s",
             K, strategy, len(files), cnt.tolist())
    if cnt.min() == 0:
        log.warning("ATA: có miền RỖNG — giảm K hoặc kiểm tra lại chỉ số cửa sổ.")
    return dom


def chronological_split(graphs, alpha=1.0, ratios=(0.7, 0.1, 0.2)):
    """Chia tập theo THỨ TỰ THỜI GIAN trong từng tệp, theo giao thức của bài báo.

    Với mỗi tệp mô phỏng: giữ lại tiền tố alpha đầu tiên theo thời gian, rồi
    cắt tuần tự thành train / val / test theo tỉ lệ 7:1:2. Nhờ vậy

        max τ(train)  <  min τ(val)  <  min τ(test)

    được bảo đảm TRONG TỪNG lần chạy mô phỏng.

    Vì sao chia trong từng tệp thay vì trên một dòng thời gian toàn cục: mỗi
    tệp RADAR là một lần chạy mô phỏng độc lập với đồng hồ riêng bắt đầu từ 0,
    nên ghép chúng lại thành một trục thời gian duy nhất là vô nghĩa về mặt
    vật lý.

    Tham số alpha là "tỉ lệ giữ lại" của bài báo: nó tách bạch ảnh hưởng của
    NGÂN SÁCH DỮ LIỆU khỏi ảnh hưởng của DỊCH CHUYỂN THỜI GIAN. Nếu hiệu năng
    tăng đơn điệu theo alpha thì kích thước mẫu là nút thắt; nếu hiệu năng đi
    ngang hoặc dao động thì dịch chuyển thời gian mới là nguyên nhân chính.
    """
    keys = [_graph_time_key(g, i) for i, g in enumerate(graphs)]
    files = {}
    for i, (f, w) in enumerate(keys):
        files.setdefault(f, []).append((w, i))

    tr, va, te = [], [], []
    for f, items in files.items():
        items.sort()
        n_keep = max(3, int(round(alpha * len(items))))
        items = items[:n_keep]
        n = len(items)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        for pos, (_w, i) in enumerate(items):
            (tr if pos < n_tr else va if pos < n_tr + n_va else te).append(graphs[i])

    log.info("Chia THEO THỜI GIAN (alpha=%.2f, 7:1:2 trong từng tệp): "
             "train=%d val=%d test=%d trên %d tệp",
             alpha, len(tr), len(va), len(te), len(files))
    return tr, va, te


def temporal_leakage_gap(train, test):
    """Chẩn đoán rò rỉ thời gian: khoảng cách thời gian gần nhất trung bình.

        δ(i) = min_{j ∈ train} |τ_i − τ_j|,     δ̄ = mean_i δ(i)

    δ̄ gần 0 nghĩa là mỗi mẫu kiểm thử được đánh giá dựa trên mẫu huấn luyện
    ở thời điểm gần như liền kề, tức trạng thái định tuyến cục bộ bị chia sẻ
    qua ranh giới chia tập. Đây chính là đại lượng mà bài báo dùng để chứng
    minh phép chia ngẫu nhiên che giấu dịch chuyển thời gian.

    Tính trong từng tệp rồi lấy trung bình có trọng số theo số đồ thị.
    """
    # Chẩn đoán chỉ có nghĩa khi đồ thị MANG thông tin thời gian thật. Nếu
    # thiếu, dùng chỉ số trong danh sách làm thay sẽ cho con số vô nghĩa (hai
    # danh sách rời nhau không khôi phục được vị trí toàn cục), nên trả NaN.
    has_t = any(getattr(g, "win_idx", None) is not None or
                getattr(g, "time_win", None) is not None
                for g in list(train)[:50] + list(test)[:50])
    if not has_t:
        return float("nan")

    def _by_file(gs):
        d = {}
        for i, g in enumerate(gs):
            f, w = _graph_time_key(g, i)
            d.setdefault(f, []).append(w)
        return d

    tr_f = _by_file(train)
    te_f = _by_file(test)
    tot, num = 0, 0
    for f, ws in te_f.items():
        if f not in tr_f:
            continue
        a = np.asarray(sorted(tr_f[f]), dtype=float)
        for w in ws:
            tot += float(np.min(np.abs(a - w)))
            num += 1
    return (tot / num) if num else float("nan")
