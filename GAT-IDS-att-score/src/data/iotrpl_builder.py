"""Builder cho bộ dữ liệu IoT-RPL 2021 (Cooja simulator).

Nguồn: "IoT-RPL 2021 Cyber Attack Dataset Based on RPL Routing for IoT" —
10 tệp CSV (0.csv … 9.csv), mỗi tệp ~1 triệu dòng, mỗi dòng là MỘT gói tin.

═══════════════════════════════════════════════════════════════════════
 KHÁC BIỆT CỐT LÕI SO VỚI RADAR — quyết định toàn bộ thiết kế builder
═══════════════════════════════════════════════════════════════════════
                        RADAR                     IoT-RPL 2021
  Cột thời gian         PHY_LAYER_START_TIME      KHÔNG CÓ
  Node nguồn            SOURCE_ID (chuỗi)         from (số nguyên 1–16)
  Node đích             DESTINATION_ID (1 node)   to = DANH SÁCH node
  Chặng vật lý          TRANSMITTER/RECEIVER      KHÔNG CÓ
  RPL_RANK              CÓ                        KHÔNG CÓ
  RPL_VERSION           CÓ (hằng số)              KHÔNG CÓ
  NEXT_HOP_IP           CÓ → suy cha DODAG        KHÔNG CÓ
  Số lớp                16                        5

 HỆ QUẢ 1 — KHÔNG CÓ THỜI GIAN.
   Không thể phân cửa sổ 5 giây như RADAR. Thay bằng cửa sổ theo SỐ DÒNG:
   mỗi `window_rows` gói liên tiếp tạo một đồ thị. Vì tệp ghi tuần tự theo
   thứ tự gói, cửa sổ theo dòng là xấp xỉ hợp lý của cửa sổ thời gian.

 HỆ QUẢ 2 — CỘT `to` LÀ DANH SÁCH (trung bình 4,23 node, tối đa 8).
   Đây là bản tin quảng bá tới nhiều láng giềng. Mỗi dòng vì thế sinh
   NHIỀU cạnh: from → to[0], from → to[1], … Đồng thời chính danh sách này
   cho ta tập LÁNG GIỀNG trực tiếp, thay cho TRANSMITTER/RECEIVER của RADAR.

 HỆ QUẢ 3 — KHÔNG CÓ RANK / VERSION / NEXT_HOP.
   Toàn bộ nhóm đặc trưng dựa trên rank của bộ RADAR (rank_vs_parent,
   diff_rank, rank_per_hop…) KHÔNG tái tạo được. Bộ đặc trưng phải xây lại
   quanh những gì có thật: loại bản tin điều khiển, cấu trúc quảng bá, và
   các trường DODAG/DIO.

 HỆ QUẢ 4 — NHÃN Ở MỨC GÓI (cột `label`).
   Node được gán nhãn theo gói của nó, tương tự chế độ pkt_label của RADAR.
   Cột `label` CHỈ dùng sinh nhãn y, TUYỆT ĐỐI không vào ma trận đặc trưng.

═══════════════════════════════════════════════════════════════════════
 BỘ ĐẶC TRƯNG: 16 node × 8 cạnh
═══════════════════════════════════════════════════════════════════════
 Thiết kế giữ NGUYÊN TẮC của bộ RADAR — đối sánh giữa điều node tự bộc lộ
 và điều láng giềng quan sát — nhưng thay các đại lượng rank bằng đại lượng
 quan sát được ở bộ này.
"""
from __future__ import annotations
import os
import glob

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..utils.common import get_logger

log = get_logger()

# 5 lớp của bộ dữ liệu; Normal luôn ở chỉ số 0 cho nhất quán với RADAR
IOTRPL_CLASSES = ["Normal", "Blackhole", "Flooding", "Version", "Rank"]
IOTRPL_CLS2IDX = {c: i for i, c in enumerate(IOTRPL_CLASSES)}
# tên cột khi tệp KHÔNG có dòng tiêu đề (chỉ 0.csv có header)
IOTRPL_COLS = ["from", "to", "frame_proto", "protocol", "control_type",
               "type_cont_messg", "DOAGID_0", "DOAGID_1", "DOAGID_2", "DOAGID_3",
               "DOAG_info", "DIO_info", "object_cont_pt", "lifetime",
               "prefix_info", "valid_lifetime", "preferred_liftime", "reserved",
               "desti_prefix_0", "desti_prefix_1", "desti_prefix_2",
               "desti_prefix_3", "label"]

N_NODE_FEATS = 16
N_EDGE_FEATS = 8
EPS = 1e-6

NODE_FEATURE_NAMES = [
    # ── Nhóm A: node tự bộc lộ (0–6) ──
    "n_sent", "n_recv", "n_dis_sent", "n_dio_sent", "n_dao_sent",
    "n_ctrl_sent", "n_data_sent",
    # ── Nhóm B: láng giềng quan sát (7–10) ──
    "n_peers_out", "n_peers_in", "bcast_size_mean", "recv_from_ctrl_ratio",
    # ── Nhóm C: độ lệch / tỉ lệ (11–15) ──
    "ctrl_data_ratio", "send_share", "dis_ratio", "dio_ratio", "deg_centrality",
]
EDGE_FEATURE_NAMES = [
    "count", "log_count", "dis_ratio", "dio_ratio", "dao_ratio",
    "data_ratio", "bcast_share", "dir_flag",
]


def _norm_txt(s):
    return str(s).strip().upper()


def _parse_to(v):
    """Cột `to` chứa danh sách node dạng '1,2,5'. Trả list[int]."""
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá cột về dạng builder dùng."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["_src"] = pd.to_numeric(df["from"], errors="coerce")
    df = df[df["_src"].notna()].copy()
    df["_src"] = df["_src"].astype(int)
    df["_dsts"] = df["to"].map(_parse_to)

    ct = df["control_type"].map(_norm_txt)
    pr = df["protocol"].map(_norm_txt)
    tm = df["type_cont_messg"].map(_norm_txt)
    # Nhận dạng loại bản tin RPL — XÁC ĐỊNH BẰNG KHẢO SÁT DỮ LIỆU THẬT:
    #   control_type='DIS' luôn đi với protocol='RPL'      -> bản tin DIS
    #   control_type='RPL' luôn đi với protocol='ICMPv6', và
    #   loại cụ thể (DIO / DAO) nằm ở cột `type_cont_messg`
    # Đếm được trên 300k dòng: DIO 148.282 · DAO 53.618 · DIS 87.741
    df["_dis"] = (ct == "DIS").astype(int)
    df["_dio"] = (tm == "DIO").astype(int)
    df["_dao"] = (tm == "DAO").astype(int)
    df["_ctrl"] = ((df["_dis"] + df["_dio"]) > 0).astype(int)
    df["_data"] = (pr == "UDP").astype(int)
    df["_lab"] = df["label"].map(lambda x: IOTRPL_CLS2IDX.get(str(x).strip(), 0))
    return df


def _build_graph(w: pd.DataFrame) -> Data | None:
    """Dựng một đồ thị từ một cửa sổ gói tin."""
    # ── tập node = nguồn ∪ mọi đích trong danh sách quảng bá ──
    ns = set(w["_src"].tolist())
    for lst in w["_dsts"]:
        ns.update(lst)
    ns = {int(a) for a in ns if a > 0}
    if len(ns) < 2:
        return None
    nl = sorted(ns)
    idx = {a: i for i, a in enumerate(nl)}
    n = len(nl)

    nf = np.zeros((n, N_NODE_FEATS), dtype=np.float32)
    yl = np.zeros(n, dtype=np.int64)

    # ── gom cạnh: mỗi dòng sinh nhiều cạnh from → to[k] ──
    pair = {}
    recv_ctrl = {a: 0 for a in nl}
    recv_all = {a: 0 for a in nl}
    peers_out = {a: set() for a in nl}
    peers_in = {a: set() for a in nl}
    bsize = {a: [] for a in nl}

    # itertuples bỏ qua tên cột bắt đầu bằng "_", nên duyệt qua mảng numpy
    a_src = w["_src"].to_numpy()
    a_dst = w["_dsts"].to_numpy(dtype=object)
    a_dis = w["_dis"].to_numpy(); a_dio = w["_dio"].to_numpy()
    a_dao = w["_dao"].to_numpy(); a_dat = w["_data"].to_numpy()
    a_ctl = w["_ctrl"].to_numpy()
    for r in range(len(a_src)):
        s = int(a_src[r]); ds = a_dst[r]
        if s not in idx:
            continue
        bsize[s].append(len(ds))
        for d in ds:
            d = int(d)
            if d not in idx or d == s:
                continue
            k = (idx[s], idx[d])
            st = pair.setdefault(k, {"c": 0, "dis": 0, "dio": 0, "dao": 0,
                                     "data": 0, "bs": 0})
            st["c"] += 1
            st["dis"] += int(a_dis[r]); st["dio"] += int(a_dio[r])
            st["dao"] += int(a_dao[r]); st["data"] += int(a_dat[r])
            st["bs"] += len(ds)
            peers_out[s].add(d); peers_in[d].add(s)
            recv_all[d] += 1
            recv_ctrl[d] += int(a_ctl[r])

    # ── đặc trưng node ──
    for a, i in idx.items():
        s_pkt = w[w["_src"] == a]
        ns_ = len(s_pkt)
        nr_ = recv_all[a]
        n_dis = int(s_pkt["_dis"].sum()); n_dio = int(s_pkt["_dio"].sum())
        n_dao = int(s_pkt["_dao"].sum()); n_dat = int(s_pkt["_data"].sum())
        n_ctl = int(s_pkt["_ctrl"].sum())

        # Nhóm A — node tự bộc lộ
        nf[i, 0] = ns_
        nf[i, 1] = nr_
        nf[i, 2] = n_dis
        nf[i, 3] = n_dio
        nf[i, 4] = n_dao
        nf[i, 5] = n_ctl
        nf[i, 6] = n_dat
        # Nhóm B — láng giềng quan sát
        nf[i, 7] = len(peers_out[a])
        nf[i, 8] = len(peers_in[a])
        nf[i, 9] = float(np.mean(bsize[a])) if bsize[a] else 0.0
        nf[i, 10] = recv_ctrl[a] / (nr_ + EPS)
        # Nhóm C — tỉ lệ / độ lệch (đều chặn trong [0,1] hoặc bị log nén)
        nf[i, 11] = n_ctl / (n_ctl + n_dat + EPS)          # ctrl_data_ratio
        # chặn trong [0,1]: dạng ns_/nr_ bùng nổ tới 2×10⁹ khi nr_=0
        nf[i, 12] = ns_ / (ns_ + nr_ + EPS)                 # send_share
        nf[i, 13] = n_dis / (ns_ + EPS)                     # dis_ratio
        nf[i, 14] = n_dio / (ns_ + EPS)                     # dio_ratio
        # HỢP hai tập, tránh đếm trùng láng giềng vừa gửi vừa nhận (từng vượt 1,0)
        nf[i, 15] = len(peers_out[a] | peers_in[a]) / (n - 1 + EPS)

        # ── NHÃN: chỉ từ cột label, KHÔNG vào đặc trưng ──
        lab = s_pkt["_lab"]
        atk = lab[lab > 0]
        if len(atk):
            yl[i] = int(atk.mode().iloc[0])

    # ── đặc trưng cạnh ──
    if not pair:
        ei = torch.arange(n).repeat(2, 1)
        ea = torch.zeros((n, N_EDGE_FEATS), dtype=torch.float32)
        return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea,
                    y=torch.tensor(yl))

    sl, dl, ef = [], [], []
    for (u, v), st in pair.items():
        c = st["c"]
        f = [float(c), float(np.log1p(c)), st["dis"] / c, st["dio"] / c,
             st["dao"] / c, st["data"] / c, st["bs"] / c / 8.0, 0.0]
        sl.append(u); dl.append(v); ef.append(f)
        sl.append(v); dl.append(u); ef.append(f[:-1] + [1.0])
    ei = torch.tensor([sl, dl], dtype=torch.long)
    ea = torch.tensor(ef, dtype=torch.float32)
    return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea, y=torch.tensor(yl))


def build_graphs_from_iotrpl_csv(df: pd.DataFrame, window_rows: int = 2000):
    """Chia một tệp thành các cửa sổ theo SỐ DÒNG rồi dựng đồ thị."""
    df = _prepare(df)
    if df.empty:
        return []
    out = []
    n_win = int(np.ceil(len(df) / window_rows))
    for w in range(n_win):
        chunk = df.iloc[w * window_rows:(w + 1) * window_rows]
        if len(chunk) < 20:
            continue
        g = _build_graph(chunk)
        if g is not None:
            out.append(g)
    return out


def load_iotrpl(csv_dir: str, window_rows: int = 2000, max_rows_per_file=None,
                skip_head: int = 300000, tag_source: bool = True):
    """Đọc mọi tệp .csv trong thư mục, trả list[Data].

    max_rows_per_file: giới hạn số dòng mỗi tệp (dùng khi chạy thử nhanh).

    skip_head: BỎ QUA n dòng đầu mỗi tệp. MẶC ĐỊNH 300.000 và đây là tham số
        BẮT BUỘC hiểu đúng:

        Kiểm chứng bằng md5 cho thấy các tệp 1,2,3,5,6,8,9 có 200.000 dòng đầu
        GIỐNG HỆT NHAU (cùng mã băm 6e06489a6ed4); điểm phân kỳ đầu tiên giữa
        1.csv và 2.csv nằm ở dòng 260.144. Nghĩa là mười tệp không phải mười
        kịch bản độc lập, mà là mười LẦN CHẠY của cùng một cấu hình mạng, chia
        sẻ chung pha khởi động.

        Nếu không bỏ phần đầu, cửa sổ đầu của tệp trong tập KIỂM THỬ sẽ trùng
        khít với cửa sổ đầu của tệp trong tập HUẤN LUYỆN — rò rỉ train→test
        thật sự. Đặt 0 để tắt (chỉ dùng khi muốn tái lập lỗi này).

    tag_source: gắn chỉ số tệp vào thuộc tính `src_file` của mỗi đồ thị, để
        `split_by_file()` chia tập theo lần chạy mô phỏng.
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not files:
        log.error("Không tìm thấy CSV trong %s", csv_dir)
        return []
    log.info("IoT-RPL: tìm thấy %d tệp | cửa sổ = %d dòng/đồ thị",
             len(files), window_rows)

    graphs = []
    for f in files:
        try:
            # Chỉ 0.csv có dòng tiêu đề; các tệp còn lại thì dòng đầu đã là dữ liệu.
            first = open(f, encoding="utf-8", errors="replace").readline()
            has_hdr = first.lower().startswith("from,to")
            df = pd.read_csv(f, low_memory=False, nrows=max_rows_per_file,
                             header=0 if has_hdr else None,
                             names=None if has_hdr else IOTRPL_COLS)
            if not has_hdr:
                df.columns = IOTRPL_COLS
            n_raw = len(df)
            if skip_head and skip_head > 0:
                df = df.iloc[skip_head:]
            if len(df) < window_rows:
                log.warning("  %-28s còn %d dòng sau khi bỏ đầu — bỏ qua tệp",
                            os.path.basename(f), len(df))
                continue
            gs = build_graphs_from_iotrpl_csv(df, window_rows)
            if tag_source:
                fid = os.path.splitext(os.path.basename(f))[0]
                for g in gs:
                    g.src_file = fid
            graphs += gs
            log.info("  %-28s %8d dòng (bỏ %d đầu) -> %5d đồ thị %s",
                     os.path.basename(f), n_raw, skip_head, len(gs),
                     "(có header)" if has_hdr else "")
        except Exception as e:
            log.warning("  Bỏ qua %s: %s", os.path.basename(f), e)

    if graphs:
        cnt = {}
        for g in graphs:
            for v, c in zip(*torch.unique(g.y, return_counts=True)):
                cnt[IOTRPL_CLASSES[int(v)]] = cnt.get(IOTRPL_CLASSES[int(v)], 0) + int(c)
        log.info("IoT-RPL: %d đồ thị | d_n=%d d_e=%d", len(graphs),
                 graphs[0].x.shape[1], graphs[0].edge_attr.shape[1])
        log.info("IoT-RPL: phân bố node theo lớp: %s", cnt)
    return graphs


def split_by_file(graphs, train_files, val_files, test_files):
    """Chia tập theo TỆP NGUỒN thay vì ngẫu nhiên.

    Mỗi đồ thị mang thuộc tính `src_file` (tên tệp không đuôi). Hàm này gom
    theo tệp, bảo đảm KHÔNG đồ thị nào của cùng một lần chạy mô phỏng xuất
    hiện ở cả tập huấn luyện lẫn tập kiểm thử.

    So với chia ngẫu nhiên, cách này khắt khe hơn nhiều: mô hình không thể
    ghi nhớ cửa sổ cụ thể đã thấy, mà phải khái quát sang lần chạy khác.

    LƯU Ý GIỚI HẠN: cách này KHÔNG giải quyết được vấn đề node tấn công cố
    định (node 12 là Blackhole ở MỌI tệp). Nó chỉ loại bỏ rò rỉ do trùng cửa
    sổ, không loại bỏ khả năng mô hình học theo danh tính node.
    """
    def norm(x):
        return {str(t).strip() for t in (x.split(",") if isinstance(x, str) else x)}

    tr_s, va_s, te_s = norm(train_files), norm(val_files), norm(test_files)
    overlap = (tr_s & te_s) | (tr_s & va_s) | (va_s & te_s)
    if overlap:
        raise ValueError("Tệp xuất hiện ở nhiều tập: %s" % sorted(overlap))

    train, val, test, unknown = [], [], [], []
    for g in graphs:
        fid = str(getattr(g, "src_file", "")).strip()
        if fid in tr_s:
            train.append(g)
        elif fid in va_s:
            val.append(g)
        elif fid in te_s:
            test.append(g)
        else:
            unknown.append(fid)
    if unknown:
        log.warning("%d đồ thị có src_file không thuộc tập nào: %s",
                    len(unknown), sorted(set(unknown))[:5])
    log.info("Chia THEO TỆP — train=%s (%d đồ thị) | val=%s (%d) | test=%s (%d)",
             sorted(tr_s), len(train), sorted(va_s), len(val),
             sorted(te_s), len(test))
    return train, val, test
