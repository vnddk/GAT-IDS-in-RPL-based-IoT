"""Chuyển bộ RPLGuard_Dataset-v1-2025 (nhật ký sự kiện gói) thành đồ thị cửa sổ.

NGUỒN
=====
RPLGuard_Dataset-v1-2025 — IEEE DataPort,
https://ieee-dataport.org/documents/rplguarddataset-v1-2025-rpl-dataset-guarding-against-attacks
Mô phỏng bằng Contiki-NG / Cooja. Theo tệp readme đi kèm bộ dữ liệu:
ba bậc mạng gồm 10, 20 và 30 nút khách, mỗi bậc có 1 nút gốc (sink),
1 nút máy chủ, và lần lượt 1, 2, 3 nút tấn công. 29 đặc trưng mỗi dòng.

Hai thư mục tấn công: Blackhole và Greyhole, mỗi thư mục 3 tệp CSV.

=====================================================================
BA PHÁT HIỆN KHI KIỂM TRA DỮ LIỆU THẬT — ĐỌC KỸ TRƯỚC KHI DÙNG
=====================================================================

(1) NHÃN NẰM Ở CỘT `node_type`, KHÔNG PHẢI `node_honest_level`.
    Giá trị quan sát được của `node_type`:
        CHILD_NODE / ROOT_NODE / EDGE_NODE   -> bình thường
        "Blackhole Attack" / "Greyhole Attack" -> tấn công
    Cột `node_honest_level` nhận giá trị 768..790 cho CẢ hai nhãn, nên
    KHÔNG dùng làm nhãn được. (Phiên bản builder trước giả định sai điều
    này; đã sửa.)

(2) `node_id` RÒ RỈ NHÃN HOÀN TOÀN.
    Nút tấn công luôn là 7, 17, 27 trong MỌI tệp, và được gán nhãn tấn
    công 100% thời gian (không có giai đoạn sạch trước khi tấn công bắt
    đầu). Luật "node_id ∈ {7,17,27}" cho F1 = 1,0000 trên cả 6 tệp.
    => node_id, From, To và mọi cột địa chỉ ĐỀU BỊ LOẠI khỏi đặc trưng.
       (Địa chỉ nhúng thẳng số hiệu nút: nút 23 -> fe80::212:7423232323.)

(3) `rank` RÒ RỈ MẠNH: giá trị 257 xuất hiện ở 94,6% dòng tấn công và
    KHÔNG BAO GIỜ ở dòng bình thường (nút thường dùng 512/768/1024).
    Gốc quyết định độ sâu 2 trên riêng cột rank cho F1 = 0,9721.
    Đây đúng kiểu thiên lệch mà nhóm RPL-Beh đã chủ động tránh bằng cách
    ngẫu nhiên hoá giá trị rank giả.
    Đo trên ĐỒ THỊ đã dựng (cửa sổ 5 s, 8.069 nút, tấn công 11,0%):

        rank_mode   F1 gốc q.định trên rank   F1 cây độ sâu 6 (mọi đặc trưng)
        ---------   -----------------------   -------------------------------
        raw                  0,8542                      0,9375
        relative             0,8269                      0,9375
        drop                   —                         0,5387

    Phép chuẩn hoá "relative" CHỈ hạ F1 từ 0,854 xuống 0,827 — KHÔNG đủ.
    Chỉ khi bỏ hẳn rank thì trần ghi nhớ mới tụt từ 0,9375 xuống 0,5387.

    VÌ SAO BỎ RANK LÀ ĐÚNG CHỨ KHÔNG PHẢI MẤT MÁT:
    hai tấn công của bộ này là Blackhole và Greyhole — tấn công VỨT GÓI,
    không phải tấn công thao túng rank. Một nút blackhole không cần giả mạo
    rank để vứt gói. Giá trị 257 vì vậy là HIỆN VẬT CỦA MÔ PHỎNG (cấu hình
    mote tấn công), không phải chữ ký nhân quả của hành vi tấn công. Tín
    hiệu nhân quả thật nằm ở drop_count, forward_pkts và etx.

    => Tham số `rank_mode`:
         "drop" (MẶC ĐỊNH)  : bỏ hẳn mọi đặc trưng dẫn xuất từ rank.
         "relative"         : giữ rank tương đối — VẪN RÒ RỈ, chỉ dùng để
                              khảo sát mức độ.
         "raw"              : giữ nguyên — CHỈ để tái hiện rò rỉ.

CÁCH DỰNG ĐỒ THỊ
================
Một cửa sổ thời gian = một đồ thị.
  Nút      : mỗi node_id xuất hiện trong cửa sổ.
  Cạnh     : cặp (From -> To) quan sát được trong cửa sổ, thêm chiều ngược.
  Nhãn nút : 0 Normal, 1 Blackhole, 2 Greyhole (hoặc nhị phân nếu gộp).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..utils.common import get_logger

log = get_logger()

RPLGUARD_CLASSES = ["Normal", "blackhole", "greyhole"]
RPLGUARD_CLASSES_BINARY = ["Normal", "attack"]

# Cột TUYỆT ĐỐI KHÔNG được dùng làm đặc trưng — xem phát hiện (2) ở trên.
LEAKY_COLUMNS = [
    "No", "node_id", "From", "To",
    "sourceaddress", "destinationaddress", "edgenodeaddress",
    "INSTANCE_ID", "Data", "node_type",
]

NODE_FEATURE_NAMES = [
    "n_events", "sending_rate", "delta_time", "received_pkts",
    "forward_pkts", "drop_count", "drop_ratio", "energy", "etx",
    "hops", "distance", "ver_diff", "honest_level",
]
NODE_FEATURE_NAMES_RANK = NODE_FEATURE_NAMES + ["rank_rel", "rank_span"]

EDGE_FEATURE_NAMES = [
    "pkt_count", "etx_mean", "hops_mean", "delta_time_mean",
    "distance_mean", "drop_diff", "fwd_ratio", "energy_diff", "is_reverse",
]


# ───────────────────────────── tiện ích ─────────────────────────────
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Lấy một cột dạng số; cột thiếu hoặc không ép được trả về 0."""
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def _parse_time(s: pd.Series) -> pd.Series:
    """Cột Time có dạng 'MM:SS.s' (ví dụ '04:59.6') -> số giây.

    Không dùng pd.to_datetime vì giá trị là ĐỘ DÀI thời gian kể từ lúc bắt
    đầu mô phỏng, không phải một mốc thời gian trong ngày.
    """
    def one(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", str(v))
        if m:
            return float(m.group(1)) * 60.0 + float(m.group(2))
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return s.map(one)


def _node_ref(s) -> pd.Series:
    """Ép một cột tham chiếu nút (`From` / `To`) về số hiệu nút.

    Cột `To` của RPLGuard dùng ký hiệu của Cooja/Wireshark: "[14 d]" nghĩa là
    nút 14, truyền trực tiếp (direct). Chỉ khoảng 6% giá trị là số thuần, nên
    phải trích chữ số đầu tiên thay vì ép kiểu trực tiếp.
    """
    if s is None:
        return pd.Series(dtype=float)
    s = pd.Series(s)
    direct = pd.to_numeric(s, errors="coerce")
    need = direct.isna()
    if need.any():
        got = s[need].astype(str).str.extract(r"(\d+)")[0]
        direct.loc[need] = pd.to_numeric(got, errors="coerce")
    return direct


def _label_of(node_type_val, binary: bool) -> int:
    """node_type -> nhãn số. Xem phát hiện (1)."""
    s = str(node_type_val).strip().lower()
    if "blackhole" in s:
        return 1
    if "greyhole" in s or "grayhole" in s:
        return 1 if binary else 2
    return 0


def _node_ids_from(df: pd.DataFrame) -> pd.Series:
    """Số hiệu nút của mỗi SỰ KIỆN.

    Dùng cột node_id để nhóm sự kiện theo nút — đây là việc CẦN THIẾT để
    dựng đồ thị. Điều bị cấm là đưa node_id vào VECTOR ĐẶC TRƯNG, chứ
    không phải dùng nó để nhóm.
    """
    return pd.to_numeric(df["node_id"], errors="coerce")


# ─────────────────────── dựng một cửa sổ ───────────────────────
def _build_one_window(w: pd.DataFrame, binary: bool, rank_mode: str):
    """Trả về một Data hoặc None nếu cửa sổ không dùng được."""
    nid = _node_ids_from(w)
    w = w.loc[nid.notna()].copy()
    if w.empty:
        return None
    w["_nid"] = nid.loc[w.index].astype(int)

    nodes = sorted(w["_nid"].unique())
    if len(nodes) < 2:
        return None
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    use_rank = rank_mode in ("relative", "raw")
    feat_names = NODE_FEATURE_NAMES_RANK if use_rank else NODE_FEATURE_NAMES
    nf = np.zeros((n, len(feat_names)), dtype=np.float32)
    ylabel = np.zeros(n, dtype=np.int64)

    # rank_min của CỬA SỔ, dùng làm mốc cho chế độ "relative".
    rk_all = _num(w, "rank")
    rk_min_win = float(rk_all[rk_all > 0].min()) if (rk_all > 0).any() else 1.0

    for node, g in w.groupby("_nid"):
        i = idx[node]
        recv = _num(g, "Received_Packets").sum()
        fwd = _num(g, "Forward_Packets").sum()
        drop = _num(g, "Drop_count").sum()
        row = [
            float(len(g)),                       # n_events
            _num(g, "Sending_rate").mean(),
            _num(g, "Delta_time").mean(),
            recv,
            fwd,
            drop,
            drop / max(recv, 1.0),               # drop_ratio
            _num(g, "Energy_consumption").mean(),
            _num(g, "etx").mean(),
            _num(g, "hops").mean(),
            _num(g, "distance").mean(),
            _num(g, "ver_diff").mean(),
            _num(g, "node_honest_level").mean(),
        ]
        if use_rank:
            rk = _num(g, "rank")
            rk = rk[rk > 0]
            if rank_mode == "raw":
                row += [float(rk.mean()) if len(rk) else 0.0,
                        float(_num(g, "rank_max").mean() - _num(g, "rank_min").mean())]
            else:   # relative — bỏ giá trị tuyệt đối, giữ tỉ lệ so với mốc cửa sổ
                row += [float(rk.mean() / rk_min_win) if len(rk) else 0.0,
                        float((_num(g, "rank_max").mean() -
                               _num(g, "rank_min").mean()) / max(rk_min_win, 1.0))]
        nf[i] = np.nan_to_num(np.asarray(row, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)

        # Nhãn nút: tấn công nếu BẤT KỲ sự kiện nào của nút trong cửa sổ
        # mang nhãn tấn công.
        labs = [_label_of(v, binary) for v in g["node_type"]]
        ylabel[i] = max(labs) if labs else 0

    # ── cạnh: cặp (From -> To) ──
    # QUAN TRỌNG: cột `To` dùng ký hiệu Cooja/Wireshark dạng "[14 d]" (nút 14,
    # truyền trực tiếp) chứ không phải số thuần. Nếu ép thẳng bằng to_numeric
    # thì 93,9% số dòng bị loại âm thầm và đồ thị gần như KHÔNG CÒN CẠNH —
    # đo được trên tệp Greyhole 30 nút: chỉ 1.215/19.939 dòng (6,1%) sinh được
    # cạnh. Trích số bằng biểu thức chính quy đưa tỉ lệ lên 100%.
    frm = _node_ref(w.get("From"))
    to = _node_ref(w.get("To"))
    pair = {}
    for k, (a, b) in enumerate(zip(frm, to)):
        if pd.isna(a) or pd.isna(b):
            continue
        a, b = int(a), int(b)
        if a not in idx or b not in idx or a == b:
            continue
        key = (idx[a], idx[b])
        st = pair.setdefault(key, {"cnt": 0, "etx": 0.0, "hops": 0.0,
                                   "dt": 0.0, "dist": 0.0, "drop": 0.0,
                                   "fwd": 0.0, "recv": 0.0, "en": 0.0})
        r = w.iloc[k]
        st["cnt"] += 1
        for fld, col in (("etx", "etx"), ("hops", "hops"), ("dt", "Delta_time"),
                         ("dist", "distance"), ("drop", "Drop_count"),
                         ("fwd", "Forward_Packets"), ("recv", "Received_Packets"),
                         ("en", "Energy_consumption")):
            try:
                st[fld] += float(pd.to_numeric(r.get(col), errors="coerce") or 0.0)
            except (TypeError, ValueError):
                pass

    # ĐỊNH DANH NÚT — bắt buộc để nối CÙNG MỘT thiết bị qua các cửa sổ liên
    # tiếp (src/models/temporal_gat.py::build_node_sequences). Thiếu trường
    # này thì hàm đó rơi về torch.arange(num_nodes), tức ghép theo VỊ TRÍ
    # trong danh sách nút đã sắp xếp cục bộ — và số nút mỗi cửa sổ của bộ
    # RPLGuard dao động mạnh (đo được: 9, 8, 4, 9, 4, 4, 9, 3, ...), nên vị
    # trí k ở cửa sổ t là một thiết bị KHÁC với vị trí k ở cửa sổ t+1. Lỗi
    # này KHÔNG gây ngoại lệ: nó lặng lẽ dựng chuỗi trộn lịch sử của nhiều
    # thiết bị. KHÔNG BAO GIỜ đưa node_ids vào nf — xem LEAKY_COLUMNS.
    _nid = torch.tensor(np.asarray(nodes, dtype=np.int64))

    ED = len(EDGE_FEATURE_NAMES)
    if not pair:
        ei = torch.arange(n).repeat(2, 1)
        ea = torch.zeros((n, ED), dtype=torch.float32)
        _g0 = Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea,
                   y=torch.tensor(ylabel))
        _g0.node_ids = _nid
        return _g0

    src, dst, ef = [], [], []
    for (a, b), st in pair.items():
        c = max(st["cnt"], 1)
        f = [float(st["cnt"]), st["etx"] / c, st["hops"] / c, st["dt"] / c,
             st["dist"] / c, st["drop"] / c,
             st["fwd"] / max(st["recv"], 1.0), st["en"] / c, 0.0]
        f = [0.0 if (np.isnan(v) or np.isinf(v)) else float(v) for v in f]
        rev = f[:-1] + [1.0]
        src.append(a); dst.append(b); ef.append(f)
        src.append(b); dst.append(a); ef.append(rev)

    _g = Data(x=torch.tensor(nf),
              edge_index=torch.tensor([src, dst], dtype=torch.long),
              edge_attr=torch.tensor(np.asarray(ef, dtype=np.float32)),
              y=torch.tensor(ylabel))
    _g.node_ids = _nid
    return _g


# ─────────────────────── dựng từ một DataFrame ───────────────────────
def build_graphs_from_df(df: pd.DataFrame, window_seconds: float = 5.0,
                         binary: bool = False, rank_mode: str = "drop"):
    """Cắt một tệp mô phỏng thành các đồ thị cửa sổ KHÔNG chồng lấn."""
    if "Time" not in df.columns:
        log.error("RPLGuard: thiếu cột Time.")
        return []
    t = _parse_time(df["Time"])
    df = df.loc[t.notna()].copy()
    if df.empty:
        return []
    t = t.loc[df.index]
    t0 = float(t.min())
    df["_win"] = ((t - t0) // float(window_seconds)).astype(int)

    graphs = []
    for wid, w in df.groupby("_win", sort=True):
        g = _build_one_window(w, binary=binary, rank_mode=rank_mode)
        if g is not None:
            g.win_idx = int(wid)
            graphs.append(g)
    return graphs


# ─────────────────────── nạp toàn bộ thư mục ───────────────────────
def load_rplguard(csv_dir: str, window_seconds: float = 5.0,
                  binary: bool = False, rank_mode: str = "drop"):
    """Nạp mọi CSV trong csv_dir (tìm đệ quy) thành danh sách đồ thị.

    Mỗi TỆP là một kịch bản mô phỏng độc lập; đồ thị mang thuộc tính
    `src_file` (chỉ số tệp), `scenario` (tên thư mục tấn công) và
    `file_name` để phép chia theo tệp mô phỏng dùng được.
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not files:
        log.error("RPLGuard: không tìm thấy CSV nào trong %s", csv_dir)
        return []

    if rank_mode not in ("relative", "drop", "raw"):
        raise ValueError("rank_mode phải là relative | drop | raw")
    if rank_mode in ("raw", "relative"):
        log.warning("rank_mode=%r: đặc trưng dẫn xuất từ rank VẪN RÒ RỈ NHÃN "
                    "(F1 gốc quyết định 0,85 với raw và 0,83 với relative; "
                    "cây độ sâu 6 đạt 0,9375 so với 0,5387 khi bỏ rank). "
                    "Điểm số thu được KHÔNG phản ánh năng lực mô hình. "
                    "Dùng rank_mode='drop' cho kết quả chính thức.", rank_mode)
    if rank_mode == "raw":
        log.warning("rank_mode='raw': giữ NGUYÊN giá trị rank tuyệt đối. "
                    "Giá trị 257 là chữ ký gần như hoàn hảo của nút tấn công "
                    "(94,6%% dòng tấn công, 0%% dòng bình thường) nên điểm số "
                    "thu được KHÔNG phản ánh năng lực mô hình. Chỉ dùng để "
                    "tái hiện rò rỉ.")

    all_graphs, n_rows = [], 0
    for fid, path in enumerate(files):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception as e:                       # pragma: no cover
            log.warning("Bỏ qua %s: %s", os.path.basename(path), e)
            continue
        if "node_type" not in df.columns:
            log.warning("Bỏ qua %s: thiếu cột node_type (cột nhãn).",
                        os.path.basename(path))
            continue
        n_rows += len(df)
        gs = build_graphs_from_df(df, window_seconds, binary, rank_mode)
        scen = os.path.basename(os.path.dirname(path))
        for g in gs:
            g.src_file = fid
            g.scenario = scen
            g.file_name = os.path.basename(path)
        all_graphs.extend(gs)
        log.info("  %-38s %6d dòng -> %4d đồ thị", os.path.basename(path),
                 len(df), len(gs))

    log.info("RPLGuard: %d tệp | %d dòng | %d đồ thị | cửa sổ %.0f s | "
             "rank_mode=%s | %s",
             len(files), n_rows, len(all_graphs), window_seconds, rank_mode,
             "nhị phân" if binary else "3 lớp")
    if all_graphs:
        log.info("RPLGuard: d_n=%d d_e=%d",
                 all_graphs[0].x.shape[1], all_graphs[0].edge_attr.shape[1])
    return all_graphs


def split_rplguard_by_file(graphs, train_ids: str, val_ids: str, test_ids: str):
    """Chia theo CHỈ SỐ TỆP mô phỏng, ví dụ '0,1,2,3:4:5'.

    Lưu ý quan trọng: với bộ này, chia theo tệp KHÔNG loại bỏ được rò rỉ
    danh tính, vì nút tấn công luôn là 7/17/27 ở MỌI tệp. Phòng tuyến duy
    nhất là không đưa node_id vào đặc trưng — builder đã làm điều đó.
    """
    def parse(s):
        return {int(x) for x in str(s).split(",") if str(x).strip() != ""}
    tr, va, te = parse(train_ids), parse(val_ids), parse(test_ids)
    out = ([], [], [])
    for g in graphs:
        f = int(getattr(g, "src_file", -1))
        if f in tr:
            out[0].append(g)
        elif f in va:
            out[1].append(g)
        elif f in te:
            out[2].append(g)
    return out


def list_rplguard_sources(graphs):
    """Thống kê số đồ thị theo tệp và theo kịch bản (để in ra khi chuyển đổi)."""
    files, scen = {}, {}
    for g in graphs:
        fn = str(getattr(g, "file_name", "?"))
        sc = str(getattr(g, "scenario", "?"))
        files[fn] = files.get(fn, 0) + 1
        scen[sc] = scen.get(sc, 0) + 1
    return scen, files
