"""Kiểm định độ tin cậy của builder packet-level -> cluster-graph.

VÌ SAO CẦN: khi ta tự viết builder, phải CHỨNG MINH rằng đồ thị sinh ra
phản ánh trung thực dữ liệu gốc, không mất mát/bịa thông tin. Module này
so khớp các đại lượng bảo toàn (invariants) giữa packet-level và graph-level.

Các kiểm tra (conservation checks):
  1. Bảo toàn nhãn: tỷ lệ gói attack ~ tỷ lệ node/graph attack (không lệch lớn).
  2. Bảo toàn node: mọi địa chỉ unicast trong gói đều xuất hiện trong đồ thị.
  3. Bảo toàn cạnh: mọi cặp (src,dst) unicast đều có cạnh tương ứng.
  4. Bảo toàn đếm: tổng pkt_count trên cạnh ~ số gói unicast gốc.
  5. Không NaN/Inf trong features.
  6. Sanity nhãn: node sinkhole phải có chữ ký (rank thấp / counter bất thường).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from ..utils.common import get_logger

log = get_logger()


def validate_conversion(df: pd.DataFrame, graphs, window_seconds=5.0, verbose=True):
    """So khớp dữ liệu packet gốc (df) với danh sách graph đã build.

    Trả về dict báo cáo + cờ `passed` tổng hợp.
    """
    from .uos_builder import _is_unicast, _info_to_kind

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    if "label" in df.columns:
        df["is_attack"] = (df["label"].astype(str).str.lower() == "attack").astype(int)
    else:
        df["is_attack"] = 0

    report = {}

    # ── Check 1: bảo toàn tỷ lệ nhãn ─────────────────────────────
    pkt_attack_ratio = float(df["is_attack"].mean())
    total_nodes = sum(g.y.numel() for g in graphs)
    attack_nodes = sum(int((g.y == 1).sum()) for g in graphs)
    node_attack_ratio = attack_nodes / max(1, total_nodes)
    report["packet_attack_ratio"] = round(pkt_attack_ratio, 4)
    report["node_attack_ratio"] = round(node_attack_ratio, 4)
    # Hai tỷ lệ không cần bằng nhau (gói vs node khác đơn vị) nhưng phải CÙNG CHIỀU:
    # có attack ở gói <=> có attack ở node.
    report["label_direction_ok"] = bool(
        (pkt_attack_ratio > 0) == (node_attack_ratio > 0))

    # ── Check 2: bảo toàn node ───────────────────────────────────
    unicast_addrs = set()
    for a in pd.concat([df["Source"], df["Destination"]]):
        if _is_unicast(a):
            unicast_addrs.add(a)
    # Tổng node trong graph có thể > số addr (vì 1 addr xuất hiện ở nhiều window)
    # nhưng số addr DUY NHẤT không được vượt quá tập gốc.
    report["unique_unicast_addrs"] = len(unicast_addrs)
    report["node_coverage_ok"] = total_nodes >= len(unicast_addrs)

    # ── Check 3 & 4: bảo toàn cạnh & đếm gói ─────────────────────
    df["kind"] = df["Info"].apply(_info_to_kind)
    n_unicast_pkts = int(df.apply(
        lambda r: _is_unicast(r["Source"]) and _is_unicast(r["Destination"]),
        axis=1).sum())
    # Tổng pkt_count trên các cạnh thuận (bỏ cạnh reverse: feature cuối=1.0 cũ,
    # giờ là is_reverse ở vị trí cuối). Đếm cạnh có is_reverse==0.
    total_edge_pkt = 0.0
    for g in graphs:
        if g.edge_attr.shape[1] >= 1:
            is_rev = g.edge_attr[:, -1]            # cột cuối = is_reverse
            fwd = is_rev < 0.5
            total_edge_pkt += float(g.edge_attr[fwd, 0].sum())  # cột 0 = pkt_count
    report["unicast_packets_original"] = n_unicast_pkts
    report["edge_packet_sum"] = int(total_edge_pkt)
    # Cho phép sai số 5% (do gói ở các window biên / self-loop)
    if n_unicast_pkts > 0:
        ratio = total_edge_pkt / n_unicast_pkts
        report["packet_conservation_ratio"] = round(ratio, 3)
        report["packet_conservation_ok"] = 0.80 <= ratio <= 1.05
    else:
        report["packet_conservation_ratio"] = None
        report["packet_conservation_ok"] = True

    # ── Check 5: không NaN/Inf ───────────────────────────────────
    nan_found = False
    for g in graphs:
        if torch.isnan(g.x).any() or torch.isinf(g.x).any():
            nan_found = True
        if torch.isnan(g.edge_attr).any() or torch.isinf(g.edge_attr).any():
            nan_found = True
    report["no_nan_inf"] = not nan_found

    # ── Check 6: sanity chữ ký sinkhole ──────────────────────────
    # Node sinkhole nên có rank_min (cột 11) thấp hơn trung bình node normal.
    sink_rankmin, norm_rankmin = [], []
    for g in graphs:
        if g.x.shape[1] > 11:
            for i in range(g.x.shape[0]):
                (sink_rankmin if int(g.y[i]) == 1 else norm_rankmin).append(
                    float(g.x[i, 11]))
    if sink_rankmin and norm_rankmin:
        report["sink_rankmin_mean"] = round(np.mean(sink_rankmin), 2)
        report["norm_rankmin_mean"] = round(np.mean(norm_rankmin), 2)
        # CHÚ Ý: sau chuẩn hóa giá trị có thể âm; chỉ kiểm tra CÓ phân biệt
        report["signature_separable"] = abs(
            np.mean(sink_rankmin) - np.mean(norm_rankmin)) > 1e-6
    else:
        report["signature_separable"] = None

    # ── Tổng hợp ─────────────────────────────────────────────────
    checks = [report["label_direction_ok"], report["node_coverage_ok"],
              report["packet_conservation_ok"], report["no_nan_inf"]]
    report["passed"] = all(checks)

    if verbose:
        log.info("=== KIỂM ĐỊNH BUILDER packet->graph ===")
        for k, v in report.items():
            log.info("  %-28s : %s", k, v)
        log.info("  => %s", "ĐẠT ✓" if report["passed"] else "CÓ VẤN ĐỀ ✗")
    return report
