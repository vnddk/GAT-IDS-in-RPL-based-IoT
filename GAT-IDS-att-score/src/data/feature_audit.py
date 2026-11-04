"""Kiểm toán rò rỉ nhãn + chọn đặc trưng LAI có attribution — cảm hứng bài báo
Amino-Acid IDS §3.4 (Chi-Squared + CNN-RF + heuristic filter attribution).

Ở đây "CNN-RF" được thay bằng "Chi2 + MutualInfo + RandomForest" phù hợp dữ
liệu bảng RPL của bạn, và Eq.(4) attribution (ánh xạ tầm quan trọng về feature
gốc) trở nên TẦM THƯỜNG vì ta chọn trực tiếp trên feature gốc (không qua CNN)
-> giữ đúng tinh thần "trace về trường mạng gốc" mà bỏ được lớp CNN thừa.

HAI CHỨC NĂNG:
  A. audit_leakage(): với mỗi feature, đo tương quan với NHÃN. Cột nào gần như
     xác định nhãn (mutual info chuẩn hoá > ngưỡng) -> cảnh báo RÒ RỈ. Rất quan
     trọng cho tính liêm chính luận văn: node feat 11/12 (lab_sent/lab_recv) và
     edge feat 6 (atk_ratio) trong radar_builder = tổng PCKT_LABEL, mà nhãn lớp
     pkt_label lại SUY TỪ chính PCKT_LABEL -> cây học tắt trên chúng.

  B. rank_features(): xếp hạng hybrid = mean(nz(Chi2), nz(MI), nz(RF_importance)),
     ánh xạ về TÊN trường RPL (giống Table 3 của bài báo). Xuất bảng cho Chương
     phương pháp + gợi ý danh sách feature nên GIỮ cho raw-skip.

Dùng độc lập (không cần torch cho phần numpy) — chỉ cần list[Data]-like có
.x (Tensor/ndarray) và .y.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.stats import chi2_contingency
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from ..utils.common import get_logger

log = get_logger()

# Tên 13 node-feature theo đúng radar_builder._bld
RADAR_NODE_FEATURES = [
    "rank_mean", "rank_min", "dio_sum", "dao_sum", "dis_sum",
    "n_sent", "n_recv", "ver_mean", "pay_mean", "col_mean",
    "total_pkts", "lab_sent", "lab_recv",
]
# Các cột NGHI rò rỉ (suy trực tiếp từ PCKT_LABEL)
RADAR_LEAKY_NODE_IDX = [11, 12]


def _stack(graphs):
    X = np.concatenate([np.asarray(g.x) for g in graphs], axis=0).astype(np.float64)
    y = np.concatenate([np.asarray(g.y) for g in graphs], axis=0).astype(np.int64)
    return X, y


def _nz(a):
    a = np.asarray(a, float)
    rng = np.ptp(a)
    return (a - a.min()) / (rng + 1e-12)


def _chi2_col(col, y, bins=12):
    if not _HAVE_SCIPY:
        return 0.0
    q = np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1])
    d = np.digitize(col, q)
    ncls = int(y.max()) + 1
    ct = np.zeros((int(d.max()) + 1, ncls))
    for a, b in zip(d, y):
        ct[a, b] += 1
    ct = ct[ct.sum(1) > 0]
    if ct.shape[0] < 2:
        return 0.0
    return float(chi2_contingency(ct + 1e-9)[0])


# ──────────────────────────────────────────────────────────────
#  A. Kiểm toán rò rỉ
# ──────────────────────────────────────────────────────────────
def audit_leakage(graphs, feature_names=None, leak_threshold=None):
    """Trả về (dict nMI, danh sách cờ đỏ).

    NGƯỠNG ĐỘNG (sửa lỗi bản v3): một đặc trưng chỉ tiết lộ "tấn công hay
    không" (oracle NHỊ PHÂN) KHÔNG THỂ đạt nMI gần 1.0 trong bài toán 16 lớp,
    vì nó không nói lớp nào. Trần lý thuyết của nó là H(nhị phân)/H(16 lớp).
    Với RADAR: H_bin=0.381, H_16=0.660 -> trần chỉ 0.578. Ngưỡng cứng 0.85 của
    bản trước vì thế KHÔNG BAO GIỜ kích hoạt -> bỏ sót rò rỉ thật.

    Bản này: (1) tính trần nhị phân, gắn cờ khi nMI >= 0.5*trần; (2) chạy thêm
    phép thử quyết định: huấn luyện gốc-cây MỘT đặc trưng để dự đoán nhãn nhị
    phân tấn công/bình thường. Nếu một cột đơn lẻ đạt F1 nhị phân > 0.95 thì
    đó là rò rỉ, bất kể nMI bao nhiêu.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score

    X, y = _stack(graphs)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    H_y = _entropy(y)
    nmi = mi / (H_y + 1e-12)

    # trần lý thuyết cho oracle nhị phân
    yb = (y != 0).astype(int)
    H_bin = _entropy(yb)
    ceil_bin = H_bin / (H_y + 1e-12)
    thr = leak_threshold if leak_threshold is not None else 0.5 * ceil_bin

    log.info("── KIỂM TOÁN RÒ RỈ NHÃN ──")
    n_cls = int(len(np.unique(y)))
    log.info("   H(%d lớp)=%.3f | H(nhị phân)=%.3f | trần oracle nhị phân=%.3f "
             "| ngưỡng cờ nMI=%.3f (tham khảo)", n_cls, H_y, H_bin, ceil_bin, thr)

    flagged = []
    for i in np.argsort(-nmi):
        # phép thử quyết định: 1 đặc trưng -> nhãn nhị phân
        stump = DecisionTreeClassifier(max_depth=2, random_state=0,
                                       class_weight="balanced")
        stump.fit(X[:, [i]], yb)
        f1b = f1_score(yb, stump.predict(X[:, [i]]), average="binary",
                       zero_division=0)
        # RÒ RỈ THẬT = một cột dự đoán nhãn gần HOÀN HẢO (bản sao nhãn).
        # nMI cao chỉ nghĩa "dự đoán tốt" — có thể là TÍN HIỆU THẬT, không phải
        # rò rỉ. Chỉ dựa vào phép thử gốc-cây 1 cột với ngưỡng cao (0.90).
        is_leak = f1b > 0.90
        tag = "  <== RÒ RỈ" if is_leak else ""
        if is_leak:
            flagged.append(names[i])
        log.info("   %-12s nMI=%.3f  F1_nhị_phân_1_cột=%.3f%s",
                 names[i], nmi[i], f1b, tag)

    if flagged:
        log.warning("Đặc trưng RÒ RỈ: %s -> PHẢI loại khỏi dữ liệu "
                    "(xem PATCH_bo_ro_ri.py), không chỉ loại khỏi raw-skip.",
                    flagged)
    else:
        log.info("Không phát hiện rò rỉ (tiêu chí quyết định: F1 gốc-cây 1 cột > 0,90).")
    return {names[i]: float(nmi[i]) for i in range(len(names))}, flagged


def _entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


# ──────────────────────────────────────────────────────────────
#  B. Xếp hạng đặc trưng lai (Chi2 + MI + RF) + attribution
# ──────────────────────────────────────────────────────────────
def rank_features(graphs, feature_names=None, exclude_idx=None, top_k=None):
    """Xếp hạng hybrid. exclude_idx: cột loại khỏi RF (vd. cột rò rỉ) để phần
    tầm quan trọng RF phản ánh tín hiệu THẬT, không bị cột rò rỉ nuốt hết."""
    X, y = _stack(graphs)
    d = X.shape[1]
    names = feature_names or [f"f{i}" for i in range(d)]
    exclude = set(exclude_idx or [])

    chi = np.array([_chi2_col(X[:, j], y) for j in range(d)])
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)

    keep = [j for j in range(d) if j not in exclude]
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(X[:, keep], y)
    imp = np.zeros(d)
    imp[keep] = rf.feature_importances_

    hybrid = (_nz(chi) + _nz(mi) + _nz(imp)) / 3.0
    order = np.argsort(-hybrid)

    log.info("── XẾP HẠNG ĐẶC TRƯNG LAI (Chi2+MI+RF) ──")
    rows = []
    for r in order:
        rows.append((names[r], float(_nz(chi)[r]), float(_nz(mi)[r]),
                     float(_nz(imp)[r]), float(hybrid[r])))
        log.info("   %-12s chi2=%.2f mi=%.2f rf=%.2f | hybrid=%.2f",
                 names[r], _nz(chi)[r], _nz(mi)[r], _nz(imp)[r], hybrid[r])
    if top_k:
        sel = [names[r] for r in order[:top_k]]
        log.info("Top-%d giữ lại: %s", top_k, sel)
    return rows


def suggest_skip_keep_idx(graphs, feature_names=None, leaky_idx=None):
    """Gợi ý danh sách chỉ số feature nên GIỮ cho raw-skip: loại cột rò rỉ."""
    d = np.asarray(graphs[0].x).shape[1]
    leaky = set(leaky_idx or [])
    keep = [i for i in range(d) if i not in leaky]
    log.info("raw-skip giữ %d/%d cột (loại rò rỉ %s)", len(keep), d, sorted(leaky))
    return keep


# Tên đặc trưng SAU KHI vá bỏ rò rỉ (xem PATCH_bo_ro_ri.py): 17 cột node
RADAR_NODE_FEATURES_CLEAN = [
    "rank_mean", "rank_min", "dio_sum", "dao_sum", "dis_sum",
    "n_sent", "n_recv", "ver_mean", "pay_mean", "col_mean", "total_pkts",
    "rank_std", "rank_range", "fwd_ratio", "ctrl_ratio", "n_peer_out", "n_peer_in",
]
RADAR_EDGE_FEATURES_CLEAN = [
    "count", "pay_avg", "dio_avg", "dao_avg", "dis_avg", "col_avg",
    "log_count", "log_pay", "dir_flag",
]


# Tên đặc trưng SAU PATCH v2 (thêm thời gian + rank tương đối): 21 cột node
RADAR_NODE_FEATURES_V2 = [
    "rank_mean", "rank_min", "dio_sum", "dao_sum", "dis_sum",
    "n_sent", "n_recv", "ver_mean", "pay_mean", "col_mean", "total_pkts",
    "rank_std", "rank_range", "fwd_ratio", "ctrl_ratio", "n_peer_out", "n_peer_in",
    "iat_mean", "iat_max", "resp_delay", "rank_vs_peer",
]


# Tên 21 đặc trưng node SẠCH (v3-clean, KHÔNG dùng PCKT_LABEL) — dùng cho RADAR
RADAR_NODE_FEATURES_CLEAN21 = [
    "rank_mean","rank_min","version_mean","dio_sent","dao_sent","dis_sent",
    "n_sent","n_recv","nbr_rank_avg","nbr_version_avg","fwd_to_others",
    "fwd_success","n_peers","diff_rank","norm_rank_diff","rank_vs_parent",
    "diff_version","ctrl_data_ratio","fwd_ratio","resp_delay","iat_std",
]


# Tên 29 đặc trưng node (v3.8: 21 sạch + 4 nhắm đích + 4 ngữ cảnh thời gian)
RADAR_NODE_FEATURES_V29 = [
    "rank_mean","rank_min","version_mean","dio_sent","dao_sent","dis_sent",
    "n_sent","n_recv","nbr_rank_avg","nbr_version_avg","fwd_to_others",
    "fwd_success","n_peers","diff_rank","norm_rank_diff","rank_vs_parent",
    "diff_version","ctrl_data_ratio","fwd_ratio","resp_delay","iat_std",
    "rank_nunique","rank_per_hop","resp_delay_median","graph_n_nodes",
    "resp_delay_ratio","iat_ratio","fwd_ratio_delta","n_nodes_ratio",
]
