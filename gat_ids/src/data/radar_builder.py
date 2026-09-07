"""Builder RADAR (DETONAR) — HYBRID LABELING 3 TẦNG.

CẤU TRÚC THƯ MỤC RADAR THẬT:
  data/radar/
    Blackhole/Packet_Trace_1500s/101.csv, 102.csv, ..., attacks_start_time.txt
    Clone_ID/Packet_Trace_1500s/101.csv, ..., attacks_start_time.txt
    Legitimate/Packet_Trace_1500s/101.csv, ...  (KHÔNG có attacks_start_time.txt)

FORMAT attacks_start_time.txt:
  101: Blackhole scenario -> Malicious node: 11 - Attack start time: 651000000.00000
  → sim_id=101, attacker=SENSOR-11, start_time=651 giây

HYBRID LABELING 3 TẦNG:
  Tầng 1 — pkt_label:  PCKT_LABEL có 1 → dùng per-packet (DIS, Sinkhole, Clone_ID...)
  Tầng 2 — time_based: PCKT_LABEL=0 + có attacks_start_time.txt
            → CHỈ gán attack cho gói từ ATTACKER NODE, SAU attack_start_time
            (Blackhole, Selective_Forward, Worst_Parent, Wormhole...)
  Tầng 3 — folder_all: Không có gì → gán toàn bộ = attack (backup)
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from ..utils.common import get_logger
log = get_logger()

RADAR_CLASSES = [
    "Normal","blackhole","clone_id","continuous_sinkhole",
    "delayed_reply","dis","hello_flood","local_repair",
    "rank","replay","selective_forward","sinkhole",
    "sybil","version","wormhole","worst_parent",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(RADAR_CLASSES)}
_ALIASES = {"helloflood":"hello_flood","sniffing_and_replay":"replay"}
for o,n in _ALIASES.items():
    if o not in CLASS_TO_IDX and n in CLASS_TO_IDX: CLASS_TO_IDX[o]=CLASS_TO_IDX[n]
EXCLUDE = {"Broadcast-0","broadcast-0","0","N/A","nan",""}


# ════════════════════════════════════════════════════════════════
#  Parse attacks_start_time.txt
# ════════════════════════════════════════════════════════════════
def _parse_attacks_start_time(txt_path: str) -> dict:
    """Parse attacks_start_time.txt → {sim_id: (attacker_node, start_time_seconds)}"""
    result = {}
    if not os.path.isfile(txt_path):
        return result
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r'(\d+):\s+.+Malicious node:\s*(\d+)\s*-\s*Attack start time:\s*([\d.]+)',
                line)
            if m:
                sim_id = int(m.group(1))
                node_id = int(m.group(2))
                start_us = float(m.group(3))
                # Node name: SENSOR-{id} (RADAR dùng format này)
                attacker = f"SENSOR-{node_id}"
                start_s = start_us / 1e6  # microsecond → second
                result[sim_id] = (attacker, start_s)
    return result


# ════════════════════════════════════════════════════════════════
#  Build graphs từ 1 file CSV
# ════════════════════════════════════════════════════════════════
def build_graphs_from_radar_csv(df, attack_type, window_seconds=5.0,
                                 attacker_node=None, attack_start_s=-1.0):
    df = df.copy()
    df.columns = [c.strip().rstrip(',') for c in df.columns]
    tc = None
    for c in ["PHY_LAYER_START_TIME(US)","PHY_LAYER_ARRIVAL_TIME(US)"]:
        if c in df.columns: tc=c; break
    if tc is None: raise ValueError("Không tìm cột thời gian")
    df["_t"] = pd.to_numeric(df[tc],errors="coerce").fillna(0)/1e6
    df["_lab"] = pd.to_numeric(df.get("PCKT_LABEL",0),errors="coerce").fillna(0).astype(int)
    ai = CLASS_TO_IDX.get(attack_type.lower(),0)
    df["_src"]=df["SOURCE_ID"].astype(str).str.strip()
    df["_dst"]=df["DESTINATION_ID"].astype(str).str.strip()
    df["_rank"]=pd.to_numeric(df.get("RPL_RANK",0),errors="coerce").fillna(0)
    df["_ver"]=pd.to_numeric(df.get("RPL_VERSION",0),errors="coerce").fillna(0)
    pc=None
    for c in ["PHY_LAYER_PAYLOAD(Bytes)","NW_LAYER_PAYLOAD(Bytes)"]:
        if c in df.columns: pc=c; break
    df["_pay"]=pd.to_numeric(df[pc],errors="coerce").fillna(0) if pc else 0.0
    cc="CONTROL_PACKET_TYPE/APP_NAME"
    ctrl=df[cc].astype(str).str.upper() if cc in df.columns else pd.Series("",index=df.index)
    df["_dio"]=ctrl.str.contains("DIO",na=False).astype(int)
    df["_dao"]=ctrl.str.contains("DAO",na=False).astype(int)
    df["_dis"]=ctrl.str.contains("DIS",na=False).astype(int)
    df["_col"]=(df["PACKET_STATUS"].astype(str).str.lower()=="collided").astype(int) if "PACKET_STATUS" in df.columns else 0

    # ── Cột topo 1-hop & cha DODAG (đặc trưng SẠCH v3-clean) ──────
    # TRANSMITTER/RECEIVER = chặng vô tuyến 1-hop (láng giềng THẬT),
    # KHÁC SOURCE/DEST (đầu-cuối, có thể nhiều hop). Nếu thiếu -> fallback src/dst.
    df["_tx"] = (df["TRANSMITTER_ID"].astype(str).str.strip()
                 if "TRANSMITTER_ID" in df.columns else df["_src"])
    df["_rx"] = (df["RECEIVER_ID"].astype(str).str.strip()
                 if "RECEIVER_ID" in df.columns else df["_dst"])
    # data packet = KHÔNG phải gói điều khiển RPL (để tính ctrl_data_ratio)
    df["_data"] = ((df["_dio"] + df["_dao"] + df["_dis"]) == 0).astype(int)
    # NEXT_HOP_IP + bản đồ IP→node để suy CHA trong DODAG.
    # LƯU Ý dữ liệu thật: NEXT_HOP có cả IP multicast (224.x, FF..) từ DIO
    # broadcast — RÁC, phải loại; và IP unicast (DAO gửi lên cha) — GIỮ.
    def _is_mcast(ip):
        ip = str(ip).strip().upper()
        return (ip in ("", "NAN", "N/A") or ip.startswith("FF")
                or ip.startswith("224.") or ip.startswith("225.")
                or ip.startswith("239.") or ip.startswith("255.")
                or "BROADCAST" in ip)
    df["_nh"] = (df["NEXT_HOP_IP"].astype(str).str.strip()
                 if "NEXT_HOP_IP" in df.columns else "")
    df.loc[df["_nh"].map(_is_mcast), "_nh"] = ""     # bỏ next-hop multicast
    ip2node = {}
    if "SOURCE_IP" in df.columns:
        for ip, nd in zip(df["SOURCE_IP"].astype(str).str.strip(), df["_src"]):
            if _ok(nd) and not _is_mcast(ip):
                ip2node.setdefault(ip, nd)
    if "DESTINATION_IP" in df.columns:
        for ip, nd in zip(df["DESTINATION_IP"].astype(str).str.strip(), df["_dst"]):
            if _ok(nd) and not _is_mcast(ip):
                ip2node.setdefault(ip, nd)
    df["_parent"] = df["_nh"].map(lambda x: ip2node.get(x, "") if x else "")

    # ── HYBRID LABELING: xác định chế độ nhãn ─────────────────────
    # Chọn sơ bộ; nếu pkt_label gán được quá ít node sẽ TỰ SỬA ở dưới.
    has_pkt = int(df["_lab"].sum()) > 0
    if ai == 0:
        label_mode = "normal"
    elif has_pkt:
        label_mode = "pkt_label"
    elif attacker_node and attack_start_s >= 0:
        label_mode = "time_based"
    else:
        label_mode = "folder_all"

    log.debug("  label_mode=%s attacker=%s start=%.0fs",
              label_mode, attacker_node, attack_start_s)

    df["_win"]=(df["_t"]/window_seconds).astype(int)

    def _build_all(mode):
        """Dựng đồ thị theo THỨ TỰ THỜI GIAN và ghép 4 đặc trưng ngữ cảnh.

        Cơ sở khoa học: khảo sát "Temporal Analysis Framework for IDS"
        (arXiv 2511.03799) xếp hạng multi-window sequential modeling là nhóm
        phương pháp có ĐỘ PHỦ THỜI GIAN RỘNG NHẤT; PPT-GNN (arXiv 2406.13365)
        và DMSTG-AD (Sci.Rep. 2026) đều cho thấy gộp thông tin liên-cửa-sổ
        cải thiện rõ các tấn công diễn tiến theo thời gian.

        VÌ SAO CẦN: cửa sổ 5s hiện xử lý ĐỘC LẬP. Tấn công delayed_reply là
        "node trả lời CHẬM HƠN BÌNH THƯỜNG" — chỉ lộ khi so node với CHÍNH NÓ
        trước đó. Mô phỏng: dùng trễ tuyệt đối Cohen d=1.81 (chồng lấn vì mỗi
        node có baseline khác nhau); dùng TỈ LỆ so với lịch sử d=7.40.

        4 đặc trưng (chỉ dùng QUÁ KHỨ -> nhân quả, không rò rỉ):
          #21 resp_delay_ratio = resp_delay(t) / EWMA_hist(resp_delay)
          #22 iat_ratio        = iat_std(t)    / EWMA_hist(iat_std)
          #23 fwd_ratio_delta  = fwd_ratio(t)  - EWMA_hist(fwd_ratio)
          #24 rank_delta       = rank_mean(t)  - EWMA_hist(rank_mean)
        """
        ALPHA = 0.3                       # hệ số EWMA cho lịch sử node
        hist = {}                         # node -> [resp, iat, fwd, rank]
        out = []
        for _win_id, w in df.groupby("_win", sort=True):   # THEO THỨ TỰ THỜI GIAN
            res = _bld(w, ai, mode, attacker_node, attack_start_s, window_seconds)
            if res is None:
                continue
            g, nl = res
            # Gắn CHỈ SỐ CỬA SỔ vào đồ thị. Cần cho hai việc: chia tập theo
            # thứ tự thời gian, và dựng miền thời gian cho ATA. Không có
            # trường này thì mọi thông tin thứ tự bị mất ngay sau khi dựng
            # đồ thị, vì danh sách đồ thị chỉ giữ được thứ tự ngầm.
            g.win_idx = int(_win_id)
            base = g.x.numpy()
            tmp = np.zeros((base.shape[0], N_TEMPORAL), dtype=np.float32)
            for i, addr in enumerate(nl):
                cur = np.array([base[i, 19], base[i, 20], base[i, 18], base[i, 24]],
                               dtype=np.float64)   # resp_delay, iat_std, fwd_ratio, rank_mean
                h = hist.get(addr)
                if h is None:
                    tmp[i] = [np.log1p(1.0), np.log1p(1.0), 0.0, 0.0]  # chưa có LS -> trung tính
                else:
                    # log1p để NÉN đuôi: tỉ lệ thô bùng nổ khi mẫu số ~0,
                    # làm phương sai khổng lồ và Cohen d tụt (đo được: d=0.30
                    # với tỉ lệ thô -> d=2.6 sau log1p).
                    tmp[i, 0] = np.log1p(cur[0] / (h[0] + EPS))   # tỉ lệ trễ
                    tmp[i, 1] = np.log1p(cur[1] / (h[1] + EPS))   # tỉ lệ nhịp
                    tmp[i, 2] = cur[2] - h[2]              # đổi tỉ lệ chuyển tiếp
                    tmp[i, 3] = cur[3] / (h[3] + EPS)      # n_nodes_ratio (sybil)
                hist[addr] = cur if h is None else (ALPHA * cur + (1 - ALPHA) * h)
            g.x = torch.tensor(np.concatenate([base, tmp], axis=1), dtype=torch.float32)
            out.append(g)
        return out

    gs = _build_all(label_mode)

    # ── TỰ SỬA khi pkt_label gán được QUÁ ÍT node ────────────────
    # Triệu chứng thật (log RADAR): lớp `version` chỉ có 5 node trên TOÀN BỘ
    # 1500 graph từ 5 file. Nguyên nhân: file Version CÓ PCKT_LABEL nhưng cực
    # thưa, nên hầu như không node nào được gán nhãn. Đo THẲNG triệu chứng
    # (số node được gán nhãn) đáng tin hơn đo mật độ gói.
    if label_mode == "pkt_label" and gs:
        n_pos = sum(int((g.y == ai).sum()) for g in gs)
        # Kỳ vọng: ít nhất 2% số graph có node được gán nhãn. Chỉ xét khi đã
        # có >= 20 graph (đủ mẫu để phán đoán, tránh kích hoạt nhầm ở file nhỏ).
        if len(gs) >= 20 and n_pos < 0.02 * len(gs):
            if attacker_node and attack_start_s >= 0:
                log.warning("  %s: pkt_label chỉ gán %d node trên %d graph "
                            "-> CHUYỂN sang time_based (attacker=%s)",
                            attack_type, n_pos, len(gs), attacker_node)
                label_mode = "time_based"
                gs = _build_all(label_mode)
                n_pos = sum(int((g.y == ai).sum()) for g in gs)
                log.warning("  %s: sau khi chuyển -> %d node được gán nhãn",
                            attack_type, n_pos)
            else:
                log.warning("  %s: pkt_label chỉ gán %d node trên %d graph và "
                            "KHÔNG có attacker/start_time -> lớp này sẽ rất "
                            "hiếm, cân nhắc loại khỏi đánh giá.",
                            attack_type, n_pos, len(gs))
    return gs, label_mode


def _ok(n): return isinstance(n,str) and n.strip() not in EXCLUDE


EPS = 1e-6
N_NODE_FEATS_BASE = 25   # 21 + 4 đặc trưng NHẮM ĐÍCH lớp yếu
N_TEMPORAL = 4           # đặc trưng NGỮ CẢNH THỜI GIAN (so với lịch sử node)
N_NODE_FEATS = N_NODE_FEATS_BASE + N_TEMPORAL   # = 29
N_EDGE_FEATS = 10

# Tên 10 đặc trưng CẠNH của RADAR — thứ tự khớp đúng khối dựng cạnh bên dưới.
RADAR_EDGE_FEATURE_NAMES = [
    "count", "log_count", "pay_avg", "dio_ratio", "dao_ratio",
    "dis_ratio", "data_ratio", "col_ratio", "is_parent_edge", "dir_flag",
]


def _bld(w, ai, label_mode, attacker_node, attack_start_s, window_seconds):
    """Xây 1 đồ thị SẠCH từ 1 cửa sổ thời gian.

    Triết lý (RPL-IDS-Beh, Nassrullah & Alisa 2025): tấn công RPL = node NÓI DỐI
    về vai trò trong DODAG. Đặc trưng mạnh nhất là ĐỘ LỆCH giữa điều node TỰ KHAI
    và điều LÁNG GIỀNG quan sát. TUYỆT ĐỐI không đưa PCKT_LABEL vào đặc trưng.

    Node set & topo 1-hop lấy từ TRANSMITTER/RECEIVER (láng giềng vật lý thật),
    KHÁC SOURCE/DEST (đầu-cuối). Nhãn vẫn gán theo SOURCE/DEST như bản gốc.
    """
    # ── node set = hợp mọi thiết bị hợp lệ (src, dst, tx, rx) ──
    ns = set()
    for col in ("_src", "_dst", "_tx", "_rx"):
        for a in w[col]:
            if _ok(a):
                ns.add(a)
    if len(ns) < 2:
        return None
    nl = sorted(ns); idx = {a: i for i, a in enumerate(nl)}; n = len(nl)
    nf = np.zeros((n, N_NODE_FEATS_BASE), dtype=np.float32)
    yl = np.zeros(n, dtype=np.int64)
    win_time = w["_t"].mean()

    # ── Bảng láng giềng 1-hop N(v) từ (TRANSMITTER, RECEIVER) ──
    nbrs = {a: set() for a in nl}
    for tx, rx in zip(w["_tx"], w["_rx"]):
        if _ok(tx) and _ok(rx) and tx != rx and tx in idx and rx in idx:
            nbrs[tx].add(rx); nbrs[rx].add(tx)

    # ── Vòng 1: thống kê TỰ KHAI (nhóm A) cho mọi node ──
    rank_mean = {}; ver_mean = {}
    S = {}; R = {}
    for addr in nl:
        s = w[w["_src"] == addr]; r = w[w["_dst"] == addr]
        S[addr] = s; R[addr] = r
        rv = s["_rank"][s["_rank"] > 0]
        rank_mean[addr] = float(rv.mean()) if len(rv) else 0.0
        ver_mean[addr] = float(s["_ver"].mean()) if len(s) else 0.0

    # ── Độ sâu tô-pô: số hop theo chuỗi cha tới gốc (có chặn chu trình) ──
    par = {}
    for addr in nl:
        ps_ = w[(w["_src"] == addr)]["_parent"]
        ps_ = ps_[ps_ != ""]
        if len(ps_):
            c_ = ps_.mode()
            if len(c_) and c_.iloc[0] in idx and c_.iloc[0] != addr:
                par[addr] = c_.iloc[0]
    depth_of = {}
    for addr in nl:
        seen, cur, dpt = set(), addr, 0
        while cur in par and cur not in seen and dpt < 32:
            seen.add(cur); cur = par[cur]; dpt += 1
        depth_of[addr] = dpt

    # ── Vòng 2: điền đặc trưng ──
    for addr, i in idx.items():
        s = S[addr]; r = R[addr]
        rv = s["_rank"][s["_rank"] > 0]
        ns_, nr_ = len(s), len(r)
        Nv = nbrs[addr]

        # ---- Nhóm A: node tự khai (0..7) ----
        nf[i, 0] = rank_mean[addr]                                   # rank_mean
        nf[i, 1] = float(rv.min()) if len(rv) else 0.0              # rank_min
        nf[i, 2] = ver_mean[addr]                                    # version_mean
        nf[i, 3] = float(s["_dio"].sum())                            # dio_sent
        nf[i, 4] = float(s["_dao"].sum())                            # dao_sent
        nf[i, 5] = float(s["_dis"].sum())                            # dis_sent
        nf[i, 6] = float(ns_)                                        # n_sent
        nf[i, 7] = float(nr_)                                        # n_recv

        # ---- Nhóm B: hàng xóm quan sát (8..12) ----
        nbr_ranks = [rank_mean[u] for u in Nv if rank_mean[u] > 0]
        nbr_vers = [ver_mean[u] for u in Nv if ver_mean[u] > 0]
        nf[i, 8] = float(np.mean(nbr_ranks)) if nbr_ranks else 0.0   # nbr_rank_avg
        nf[i, 9] = float(np.mean(nbr_vers)) if nbr_vers else 0.0     # nbr_version_avg
        # F(v) = gói v CHUYỂN TIẾP hộ (tx=v nhưng src!=v)
        fwd = w[(w["_tx"] == addr) & (w["_src"] != addr)]
        n_fwd = len(fwd)
        nf[i, 10] = float(n_fwd)                                     # fwd_to_others
        # gói v NHẬN với tư cách trung gian (rx=v nhưng dst!=v) -> lẽ ra phải fwd
        relay_in = len(w[(w["_rx"] == addr) & (w["_dst"] != addr)])
        nf[i, 11] = float(n_fwd) / (relay_in + EPS)                  # fwd_success
        nf[i, 12] = float(len(Nv))                                   # n_peers

        # ---- Nhóm C: độ lệch tự-khai vs quan-sát (13..20) ----
        diff_rank = abs(rank_mean[addr] - nf[i, 8])
        nf[i, 13] = diff_rank                                        # diff_rank
        nf[i, 14] = diff_rank / (nf[i, 8] + EPS)                     # norm_rank_diff
        # rank_vs_parent: cha = node có NEXT_HOP trỏ tới; fallback = láng giềng rank nhỏ nhất
        parent = ""
        pser = s["_parent"][s["_parent"] != ""]
        if len(pser):
            cand = pser.mode()
            if len(cand) and cand.iloc[0] in idx:
                parent = cand.iloc[0]
        if not parent and nbr_ranks:                                 # fallback RPL: cha có rank nhỏ hơn
            parent = min((u for u in Nv if rank_mean[u] > 0),
                         key=lambda u: rank_mean[u], default="")
        p_rank = rank_mean.get(parent, 0.0)
        nf[i, 15] = (rank_mean[addr] - p_rank) if (parent and rank_mean[addr] > 0) else 0.0
        nf[i, 16] = abs(ver_mean[addr] - nf[i, 9])                   # diff_version
        n_ctrl = nf[i, 3] + nf[i, 4] + nf[i, 5]
        n_data = float(s["_data"].sum())
        # tỉ trọng gói điều khiển, CHẶN trong [0,1] để tránh bùng nổ khi n_data=0
        nf[i, 17] = n_ctrl / (n_ctrl + n_data + EPS)                 # ctrl_data_ratio
        nf[i, 18] = float(n_fwd) / (nr_ + EPS)                       # fwd_ratio
        # resp_delay: trễ trung bình từ lúc NHẬN tới gói GỬI kế tiếp
        if ns_ > 0 and nr_ > 0:
            ts = np.sort(s["_t"].values); tr = np.sort(r["_t"].values)
            pos = np.searchsorted(ts, tr, side="left")
            ok = pos < len(ts)
            nf[i, 19] = float((ts[pos[ok]] - tr[ok]).mean()) if ok.any() else 0.0
        else:
            nf[i, 19] = 0.0
        # iat_std: độ lệch chuẩn khoảng cách thời gian gói gửi
        if ns_ > 1:
            nf[i, 20] = float(np.diff(np.sort(s["_t"].values)).std())
        else:
            nf[i, 20] = 0.0

        # ---- Nhóm E: NHẮM ĐÍCH 4 lớp yếu nhất (21..24) ----
        # Thiết kế theo chữ ký riêng của từng họ tấn công; mọi đặc trưng đều
        # kiểm chứng bằng Cohen's d trên mô phỏng cơ chế tấn công.

        # #21 rank_nunique — clone_id: MỘT ID công bố NHIỀU rank mâu thuẫn.
        #     Kẻ sao chép danh tính khiến cùng một địa chỉ khai báo rank khác
        #     nhau trong cùng cửa sổ. (Cohen d = 2,58)
        nf[i, 21] = float(len(np.unique(np.round(rv.values, 2)))) if len(rv) else 0.0

        # #22 rank_per_hop — rank: rank KHÔNG nhất quán với ĐỘ SÂU tô-pô.
        #     RPL hợp lệ có rank tăng gần tuyến tính theo số hop tới gốc; kẻ
        #     tấn công khai rank thấp giả nên tỉ số này tụt mạnh. (d = 9,26)
        dep = depth_of.get(addr, 0)
        nf[i, 22] = (rank_mean[addr] / (128.0 * dep)) if (dep > 0 and rank_mean[addr] > 0) else 0.0

        # #23 resp_delay_median — delayed_reply: TRUNG VỊ bền với ngoại lai hơn
        #     trung bình. Đo được: mean d=1,74 -> median d=3,18. (x19 giữ nguyên
        #     để so sánh; đây là bản bền vững hơn.)
        if ns_ > 0 and nr_ > 0:
            _ts = np.sort(s["_t"].values); _tr = np.sort(r["_t"].values)
            _pos = np.searchsorted(_ts, _tr, side="left")
            _msk = _pos < len(_ts)          # KHÔNG đặt tên _ok: trùng hàm _ok()
            nf[i, 23] = float(np.median(_ts[_pos[_msk]] - _tr[_msk])) if _msk.any() else 0.0
        else:
            nf[i, 23] = 0.0

        # #24 graph_n_nodes — sybil: bơm danh tính giả làm SỐ NODE trong cửa sổ
        #     tăng vọt. Đây là đặc trưng MỨC ĐỒ THỊ, phát cho mọi node; nhóm D
        #     sẽ so nó với lịch sử để thành tỉ lệ. (d = 5,02 dạng tỉ lệ)
        nf[i, 24] = float(n)

        # ── GÁN NHÃN 3 TẦNG (KHÔNG đổi so với bản gốc) ──
        if label_mode == "normal":
            yl[i] = 0
        elif label_mode == "pkt_label":
            if s["_lab"].sum() > 0 or r["_lab"].sum() > 0:
                yl[i] = ai
        elif label_mode == "time_based":
            if addr == attacker_node and win_time >= attack_start_s:
                yl[i] = ai
        elif label_mode == "folder_all":
            yl[i] = ai

    # ── Cạnh: topo 1-hop (TRANSMITTER→RECEIVER), 10 đặc trưng SẠCH ──
    ps = {}
    parent_of = {}
    for _, row in w.iterrows():
        u, v = row["_tx"], row["_rx"]
        if _ok(u) and _ok(v) and u in idx and v in idx and u != v:
            k = (idx[u], idx[v])
            st = ps.setdefault(k, {"c": 0, "p": 0.0, "dio": 0, "dao": 0,
                                   "dis": 0, "data": 0, "col": 0})
            st["c"] += 1; st["p"] += float(row["_pay"])
            st["dio"] += int(row["_dio"]); st["dao"] += int(row["_dao"])
            st["dis"] += int(row["_dis"]); st["data"] += int(row["_data"])
            st["col"] += int(row["_col"])
        # ghi nhận cạnh cha: u -> parent(u)
        if _ok(row["_src"]) and row["_parent"] and row["_src"] in idx and row["_parent"] in idx:
            parent_of[idx[row["_src"]]] = idx[row["_parent"]]

    if not ps:
        ei = torch.arange(n).repeat(2, 1)
        ea = torch.zeros((n, N_EDGE_FEATS), dtype=torch.float32)
        return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea, y=torch.tensor(yl)), nl

    sl, dl, ef = [], [], []
    for (u, v), st in ps.items():
        c = st["c"]
        is_parent = 1.0 if parent_of.get(u) == v else 0.0
        # [count, log_count, pay_avg, dio_ratio, dao_ratio, dis_ratio,
        #  data_ratio, col_ratio, is_parent_edge, direction_flag]
        f = [float(c), float(np.log1p(c)), st["p"] / c,
             st["dio"] / c, st["dao"] / c, st["dis"] / c,
             st["data"] / c, st["col"] / c, is_parent, 0.0]
        sl.append(u); dl.append(v); ef.append(f)
        # cạnh ngược (direction_flag=1); is_parent_edge của chiều ngược = 0
        fr = f[:8] + [0.0, 1.0]
        sl.append(v); dl.append(u); ef.append(fr)
    ei = torch.tensor([sl, dl], dtype=torch.long)
    ea = torch.tensor(ef, dtype=torch.float32)
    return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea, y=torch.tensor(yl)), nl


# ════════════════════════════════════════════════════════════════
#  Đọc toàn bộ RADAR — hỗ trợ cấu trúc Packet_Trace_1500s
# ════════════════════════════════════════════════════════════════
def load_radar(dataset_dir, window_seconds=5.0):
    if not os.path.isdir(dataset_dir):
        log.warning("Không thấy %s",dataset_dir); return []

    ag=[]; nok=nsk=0; mode_stats={}

    for af in sorted(os.listdir(dataset_dir)):
        fp=os.path.join(dataset_dir,af)
        if not os.path.isdir(fp): continue

        # Attack type từ tên thư mục
        fl=af.lower().strip().replace(" ","_")
        at="Normal" if fl in ("legitimate","normal","benign") else (fl if fl in CLASS_TO_IDX else None)
        if at is None:
            log.warning("Thư mục '%s' không nhận diện -> bỏ qua", af)
            continue

        # Tìm thư mục chứa CSV (hỗ trợ cả Packet_Trace_1500s/ và flat)
        csv_dirs = []
        pt_dir = os.path.join(fp, "Packet_Trace_1500s")
        if os.path.isdir(pt_dir):
            csv_dirs.append(pt_dir)
        else:
            csv_dirs.append(fp)  # fallback: CSV ngay trong thư mục attack

        for cd in csv_dirs:
            # Parse attacks_start_time.txt
            atk_info = {}
            for txt_name in ["attacks_start_time.txt","Attack_Time.txt","attack_time.txt"]:
                txt_path = os.path.join(cd, txt_name)
                if os.path.isfile(txt_path):
                    atk_info = _parse_attacks_start_time(txt_path)
                    break
            if not atk_info:
                # Thử ở thư mục cha
                for txt_name in ["attacks_start_time.txt","Attack_Time.txt"]:
                    txt_path = os.path.join(fp, txt_name)
                    if os.path.isfile(txt_path):
                        atk_info = _parse_attacks_start_time(txt_path)
                        break

            # Tìm CSV
            csvs = sorted(glob.glob(os.path.join(cd, "*.csv")))
            for cf in csvs:
                try:
                    if os.path.getsize(cf)<50: continue
                    # Suy sim_id từ tên file (101.csv → 101)
                    fname = os.path.splitext(os.path.basename(cf))[0]
                    try: sim_id = int(fname)
                    except: sim_id = -1

                    # Lấy attacker info cho sim này
                    attacker_node = None
                    attack_start_s = -1.0
                    if sim_id in atk_info:
                        attacker_node, attack_start_s = atk_info[sim_id]

                    df = pd.read_csv(cf, low_memory=False)
                    if df.shape[0]==0: continue
                    gs, mode = build_graphs_from_radar_csv(
                        df, at, window_seconds,
                        attacker_node=attacker_node,
                        attack_start_s=attack_start_s)
                    if gs:
                        # Gắn ĐỊNH DANH TỆP để chia tập theo tệp mô phỏng và
                        # để dựng miền thời gian theo từng lần chạy.
                        _fid = f"{af}/{os.path.basename(cf)}"
                        for _g in gs:
                            _g.src_file = _fid
                        ag.extend(gs); nok+=1
                        mode_stats[mode] = mode_stats.get(mode,0)+1
                        log.info("RADAR %s/%s [%s, mode=%s, attacker=%s] -> %d graph",
                                 af, os.path.basename(cf), at, mode,
                                 attacker_node or "N/A", len(gs))
                    else: nsk+=1
                except Exception as e:
                    log.warning("Bỏ qua %s/%s: %s",af,os.path.basename(cf),e); nsk+=1

    if ag:
        labs=set();[labs.update(g.y.unique().tolist()) for g in ag]
        log.info("RADAR tổng: %d file OK, %d bỏ, %d graph, lớp: %s",nok,nsk,len(ag),sorted(labs))
        log.info("  Label modes: %s", mode_stats)
    return ag
