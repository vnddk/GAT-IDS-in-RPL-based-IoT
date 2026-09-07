"""Chuyển dữ liệu RPLGuard (node-level event log) -> cluster-graph.

VÌ SAO RPLGUARD TỐT HƠN UOS CHO ĐỀ TÀI:
  RPLGuard có SẴN các đặc trưng RPL giàu mà UOS thiếu:
  ETX, hops, Drop_count, Forward_Packets, rank_min/max, ver_diff, energy.
  Đây chính xác là tín hiệu edge-aware GAT cần.

CỘT GỐC CỦA RPLGUARD (29 features):
  No, Time, From, To, sourceaddress, destinationaddress, edgenodeaddress,
  INSTANCE_ID, rank, rank_min, rank_max, node_id, ver, ver_min, ver_max,
  ver_diff, Sending_time, Sending_rate, Delta_time, Received_Packets,
  Forward_Packets, Drop_count, Energy_consumption, etx, hops, distance,
  node_type, node_honest_level, Data

CÁCH DỰNG ĐỒ THỊ (1 cửa sổ thời gian = 1 graph):
  - Node      : mỗi node_id duy nhất là 1 nút.
  - Node feat : rank, rank_min/max, ver/ver_diff, ETX, hops, energy,
                drop_count, forward_packets, sending_rate, received_packets (d_n=13).
  - Edge      : cặp (From -> To) quan sát trong cửa sổ.
  - Edge feat : thống kê cặp đó: ETX_mean, hops_mean, drop_diff, fwd_ratio,
                delta_time_mean, distance_mean, pkt_count, energy_diff (d_e=8).
  - Nhãn node : từ node_honest_level (0 = honest/normal; > 0 = attacker/anomaly).

Tấn công: Blackhole + Grayhole (multi-class nếu phân biệt, hoặc binary attack/normal).
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..utils.common import get_logger

log = get_logger()

# Nhãn: dựa trên node_honest_level
# honest_level=0 (hoặc tương đương) = Normal; >0 = Attack
# Nếu dataset phân biệt Blackhole vs Grayhole bằng giá trị khác, đổi mapping ở đây.
RPLGUARD_CLASSES = ["Normal", "Attack"]

# Tên cột chuẩn hóa (strip whitespace)
_COL_REQUIRED = ["Time", "node_id"]
_COL_NUM = [
    "rank", "rank_min", "rank_max", "ver", "ver_min", "ver_max", "ver_diff",
    "Sending_rate", "Delta_time", "Received_Packets", "Forward_Packets",
    "Drop_count", "Energy_consumption", "etx", "hops", "distance",
]

NODE_FEATURE_NAMES = [
    "rank", "rank_min", "rank_max", "ver_diff", "etx", "hops",
    "energy", "drop_count", "fwd_packets", "recv_packets",
    "sending_rate", "distance", "delta_time_mean",
]

EDGE_FEATURE_NAMES = [
    "pkt_count", "etx_mean", "hops_mean", "drop_diff",
    "fwd_ratio", "delta_time_mean", "distance_mean", "is_reverse",
]


def _safe_num(df, col):
    """Chuyển cột sang numeric, điền 0 nếu thiếu hoặc lỗi."""
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _determine_label(row_or_series) -> int:
    """Suy nhãn từ node_honest_level. 0=Normal, 1=Attack."""
    val = row_or_series
    if isinstance(val, pd.Series):
        val = val.mean()
    try:
        return 0 if float(val) == 0 else 1
    except (ValueError, TypeError):
        # Nếu là chuỗi: "honest" -> 0, bất kỳ thứ gì khác -> 1
        s = str(val).strip().lower()
        return 0 if s in ("0", "honest", "normal", "benign", "") else 1


def build_graphs_from_df(df: pd.DataFrame, window_seconds: float = 10.0):
    """Dựng list[Data] từ DataFrame RPLGuard."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Xác nhận cột bắt buộc
    for c in _COL_REQUIRED:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc '{c}'. Cột hiện có: {list(df.columns)}")

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    for c in _COL_NUM:
        df[c] = _safe_num(df, c)

    # Tìm cột From/To (cặp gửi-nhận) — RPLGuard dùng "From", "To"
    from_col = "From" if "From" in df.columns else "sourceaddress"
    to_col = "To" if "To" in df.columns else "destinationaddress"

    df["win"] = (df["Time"] // window_seconds).astype(int)

    graphs = []
    for _, w in df.groupby("win"):
        g = _build_one_window(w, from_col, to_col)
        if g is not None:
            graphs.append(g)
    return graphs


def _build_one_window(w: pd.DataFrame, from_col: str, to_col: str):
    # Tập nút = mọi node_id duy nhất
    node_ids = sorted(w["node_id"].dropna().unique())
    if len(node_ids) < 2:
        return None
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    # ── Node features (d_n = 13) ─────────────────────────────────
    nf = np.zeros((n, 13), dtype=np.float32)
    ylabel = np.zeros(n, dtype=np.int64)

    for nid, i in idx.items():
        rows = w[w["node_id"] == nid]
        nf[i, 0] = rows["rank"].mean()
        nf[i, 1] = rows["rank_min"].mean()
        nf[i, 2] = rows["rank_max"].mean()
        nf[i, 3] = rows["ver_diff"].mean()
        nf[i, 4] = rows["etx"].mean()
        nf[i, 5] = rows["hops"].mean()
        nf[i, 6] = rows["Energy_consumption"].mean()
        nf[i, 7] = rows["Drop_count"].mean()         # dấu hiệu blackhole/grayhole
        nf[i, 8] = rows["Forward_Packets"].mean()
        nf[i, 9] = rows["Received_Packets"].mean()
        nf[i, 10] = rows["Sending_rate"].mean()
        nf[i, 11] = rows["distance"].mean()
        nf[i, 12] = rows["Delta_time"].mean()

        # Nhãn: từ node_honest_level
        if "node_honest_level" in rows.columns:
            ylabel[i] = _determine_label(rows["node_honest_level"].iloc[0])

    # ── Edges: cặp (From -> To) ─────────────────────────────────
    pair_stats = {}
    for _, row in w.iterrows():
        s_raw = row.get(from_col)
        d_raw = row.get(to_col)
        # map From/To sang node_id; nếu From/To là node_id trực tiếp
        s_id = s_raw if s_raw in idx else None
        d_id = d_raw if d_raw in idx else None
        if s_id is None or d_id is None:
            continue
        key = (idx[s_id], idx[d_id])
        if key[0] == key[1]:
            continue    # bỏ self-loop ở bước build
        st = pair_stats.setdefault(key, {
            "cnt": 0, "etx": 0.0, "hops": 0.0, "drop": 0.0,
            "fwd": 0.0, "delta": 0.0, "dist": 0.0,
        })
        st["cnt"] += 1
        st["etx"] += float(row.get("etx", 0))
        st["hops"] += float(row.get("hops", 0))
        st["drop"] += float(row.get("Drop_count", 0))
        st["fwd"] += float(row.get("Forward_Packets", 0))
        st["delta"] += float(row.get("Delta_time", 0))
        st["dist"] += float(row.get("distance", 0))

    EDGE_DIM = 8
    if not pair_stats:
        # fallback self-loop
        ei = torch.arange(n).repeat(2, 1)
        ea = torch.zeros((n, EDGE_DIM), dtype=torch.float32)
        return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea,
                    y=torch.tensor(ylabel))

    src, dst, edge_feat = [], [], []
    for (s, d), st in pair_stats.items():
        cnt = max(1, st["cnt"])
        recv = nf[d, 9]     # recv_packets của nút đích
        feat = [
            cnt,
            st["etx"] / cnt,
            st["hops"] / cnt,
            st["drop"] / cnt,                               # drop trên cạnh = blackhole
            st["fwd"] / max(1, recv) if recv > 0 else 0.0,  # forward ratio
            st["delta"] / cnt,
            st["dist"] / cnt,
            0.0,    # is_reverse
        ]
        rev = feat[:-1] + [1.0]
        src.append(s); dst.append(d); edge_feat.append(feat)
        src.append(d); dst.append(s); edge_feat.append(rev)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat, dtype=torch.float32)
    return Data(x=torch.tensor(nf), edge_index=edge_index,
                edge_attr=edge_attr, y=torch.tensor(ylabel))


def load_rplguard_csv(csv_dir: str, window_seconds: float = 10.0):
    """Đọc mọi CSV RPLGuard trong thư mục -> list[Data]."""
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return []
    all_graphs = []
    n_ok, n_skip = 0, 0
    for f in files:
        try:
            if os.path.getsize(f) < 16:
                log.warning("Bỏ qua (rỗng): %s", os.path.basename(f))
                n_skip += 1
                continue
            df = pd.read_csv(f)
            if df.shape[0] == 0:
                n_skip += 1
                continue
            gs = build_graphs_from_df(df, window_seconds)
            all_graphs.extend(gs)
            n_ok += 1
            log.info("RPLGuard %s -> %d graph", os.path.basename(f), len(gs))
        except Exception as e:
            log.warning("Bỏ qua %s: %s", os.path.basename(f), e)
            n_skip += 1
    log.info("RPLGuard tổng: %d file OK, %d bỏ qua, %d graph",
             n_ok, n_skip, len(all_graphs))
    return all_graphs
