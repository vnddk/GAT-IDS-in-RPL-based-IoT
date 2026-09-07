"""Chuyển dữ liệu packet-level UOS_IOTSH_2024 (Wireshark export) -> cluster-graph.

VÌ SAO CẦN FILE NÀY:
  Dataset UOS là packet-level (mỗi dòng = 1 gói tin), trong khi Edge-aware GAT
  cần đồ thị (node = thiết bị, edge = liên kết, có node/edge features). File này
  làm "cầu nối": gom gói theo cửa sổ thời gian -> dựng đồ thị DODAG cho mỗi cửa sổ.

CỘT GỐC CỦA UOS:
  No., Time, Source, Destination, Protocol, Length, Rank, Info,
  Interval, RDAO, RDIO, SDIO, SDAO, label   (label: normal/attack — chỉ sinkhole)

CÁCH DỰNG ĐỒ THỊ (1 cửa sổ thời gian = 1 graph):
  - Node      : mỗi địa chỉ Source/Destination unicast là 1 nút.
  - Node feat : thống kê gói của nút trong cửa sổ (Rank, đếm DIO/DAO/DIS,
                Interval, Length, RDAO/RDIO/SDIO/SDAO trung bình...).
  - Edge      : cặp (Source -> Destination) unicast quan sát được trong cửa sổ.
  - Edge feat : thống kê trên cặp đó (số gói, Length tb, Interval tb...).
  - Nhãn node : 'attack' nếu nút có tham gia (gửi/nhận) gói attack trong cửa sổ.

Nhãn ở đây là NHỊ PHÂN {Normal=0, Sinkhole=1} — khớp đặc thù UOS (chỉ sinkhole).
Khi ghép thêm ROUT-4/IoT-RPL cho đa lớp, mở rộng ánh xạ nhãn ở bước hợp nhất.
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

# Nhãn nhị phân cho UOS (chỉ sinkhole)
UOS_CLASSES = ["Normal", "Sinkhole"]

# Tên 13 đặc trưng NODE — thứ tự khớp đúng các phép gán nf[i, k] bên dưới.
UOS_NODE_FEATURE_NAMES = [
    "rank_mean", "dio_sent", "dao_sent", "dis_sent", "n_sent", "n_recv",
    "interval_mean", "rdao_mean", "rdio_mean", "sdio_mean", "sdao_mean",
    "rank_min", "length_mean",
]
# Tên 10 đặc trưng CẠNH — thứ tự khớp khối dựng cạnh bên dưới.
UOS_EDGE_FEATURE_NAMES = [
    "pkt_count", "len_mean", "interval_mean", "rdao_mean", "rdio_mean",
    "sdio_mean", "sdao_mean", "dio_ratio", "dao_ratio", "is_reverse",
]

MULTICAST_PREFIXES = ("ff02", "ff03", "ff05")   # địa chỉ multicast IPv6 -> bỏ khi làm node


def _is_unicast(addr: str) -> bool:
    if not isinstance(addr, str) or not addr:
        return False
    return not addr.lower().startswith(MULTICAST_PREFIXES)


def _info_to_kind(info: str) -> str:
    """Phân loại bản tin RPL từ cột Info."""
    if not isinstance(info, str):
        return "other"
    if "Solicitation" in info:        # DIS
        return "dis"
    if "Information Object" in info:  # DIO (mang Rank)
        return "dio"
    if "Advertisement" in info:       # DAO
        return "dao"
    return "other"


def build_graphs_from_df(df: pd.DataFrame, window_seconds: float = 10.0,
                         default_label: str = "normal"):
    """Dựng list[Data] từ một DataFrame UOS.

    CHỐNG CHỊU SCHEMA KHÔNG ĐỒNG NHẤT (thực tế UOS có nhiều biến thể header):
      - Cột bắt buộc: Time, Source, Destination, Info. Thiếu -> báo lỗi rõ.
      - Cột tùy chọn (Rank, Length, Interval, RDAO/RDIO/SDIO/SDAO): thiếu -> điền 0.
      - Cột 'label' có thể KHÔNG tồn tại. QUY TẮC: chỉ tin cột label trong CSV;
        nếu không có cột label -> toàn bộ gán Normal (KHÔNG suy từ tên file).
        (Tham số default_label giữ lại cho tương thích nhưng không dùng cho nhãn.)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["Time", "Source", "Destination", "Info"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc {missing}")

    # Cột tùy chọn -> điền 0 nếu thiếu
    for col in ["Length", "Interval", "RDAO", "RDIO", "SDIO", "SDAO"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    if "Rank" not in df.columns:
        df["Rank"] = np.nan
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    df["kind"] = df["Info"].apply(_info_to_kind)

    # Nhãn: CHỈ tin vào cột 'label' trong CSV.
    # Nếu KHÔNG có cột label -> mặc định toàn bộ là Normal (theo yêu cầu),
    # KHÔNG suy từ tên file (tránh gán nhãn rác cho nút không phải attacker).
    if "label" in df.columns:
        df["is_attack"] = (df["label"].astype(str).str.lower() == "attack").astype(int)
    else:
        df["is_attack"] = 0

    df["win"] = (df["Time"] // window_seconds).astype(int)

    graphs = []
    for _win_id, w in df.groupby("win", sort=True):     # THEO THỨ TỰ THỜI GIAN
        g = _build_one_window(w)
        if g is not None:
            # Gắn CHỈ SỐ CỬA SỔ. Cần cho hai việc: chia tập theo thứ tự thời
            # gian (--split chrono) và dựng miền thời gian cho ATA. Không có
            # trường này, hàm chẩn đoán rò rỉ sẽ trả NaN và ATA phải rơi về
            # dùng chỉ số trong danh sách — vốn không phải thời gian thật.
            g.win_idx = int(_win_id)
            graphs.append(g)
    return graphs


def _build_one_window(w: pd.DataFrame):
    # Tập nút = các địa chỉ unicast xuất hiện ở Source hoặc Destination
    nodes = set()
    for addr in pd.concat([w["Source"], w["Destination"]]):
        if _is_unicast(addr):
            nodes.add(addr)
    if len(nodes) < 2:
        return None
    node_list = sorted(nodes)
    idx = {a: i for i, a in enumerate(node_list)}
    n = len(node_list)

    # ── Node features (d_n = 13) ─────────────────────────────────
    # [rank_mean, dio_cnt, dao_cnt, dis_cnt, pkt_sent, pkt_recv,
    #  interval_mean, rdao_mean, rdio_mean, sdio_mean, sdao_mean,
    #  rank_min, len_mean]
    nf = np.zeros((n, 13), dtype=np.float32)
    ylabel = np.zeros(n, dtype=np.int64)

    for addr, i in idx.items():
        sent = w[w["Source"] == addr]
        recv = w[w["Destination"] == addr]
        rank_vals = sent["Rank"].dropna()
        nf[i, 0] = rank_vals.mean() if len(rank_vals) else 0.0
        nf[i, 1] = (sent["kind"] == "dio").sum()
        nf[i, 2] = (sent["kind"] == "dao").sum()
        nf[i, 3] = (sent["kind"] == "dis").sum()
        nf[i, 4] = len(sent)
        nf[i, 5] = len(recv)
        nf[i, 6] = sent["Interval"].mean() if len(sent) else 0.0
        nf[i, 7] = sent["RDAO"].mean() if len(sent) else 0.0
        nf[i, 8] = sent["RDIO"].mean() if len(sent) else 0.0
        nf[i, 9] = sent["SDIO"].mean() if len(sent) else 0.0
        nf[i, 10] = sent["SDAO"].mean() if len(sent) else 0.0
        nf[i, 11] = rank_vals.min() if len(rank_vals) else 0.0   # rank thấp bất thường = sinkhole
        nf[i, 12] = sent["Length"].mean() if len(sent) else 0.0
        if sent["is_attack"].sum() > 0 or recv["is_attack"].sum() > 0:
            ylabel[i] = 1   # Sinkhole

    # ── Edges: cặp (Source->Destination) unicast ─────────────────
    # Làm giàu edge features bằng counter RPL (RDAO/RDIO/SDIO/SDAO) — tín hiệu
    # mạnh cho sinkhole: nút hút traffic có RDAO/SDIO bất thường trên cạnh của nó.
    pair_stats = {}
    for _, row in w.iterrows():
        s, d = row["Source"], row["Destination"]
        if _is_unicast(s) and _is_unicast(d) and s in idx and d in idx:
            key = (idx[s], idx[d])
            st = pair_stats.setdefault(key, {
                "cnt": 0, "len": 0.0, "intv": 0.0,
                "rdao": 0.0, "rdio": 0.0, "sdio": 0.0, "sdao": 0.0,
                "dio": 0, "dao": 0,
            })
            st["cnt"] += 1
            st["len"] += float(row["Length"])
            st["intv"] += float(row["Interval"])
            st["rdao"] += float(row["RDAO"])
            st["rdio"] += float(row["RDIO"])
            st["sdio"] += float(row["SDIO"])
            st["sdao"] += float(row["SDAO"])
            st["dio"] += int(row["kind"] == "dio")
            st["dao"] += int(row["kind"] == "dao")

    EDGE_DIM = 10
    if not pair_stats:
        ei = torch.arange(n).repeat(2, 1)
        ea = torch.zeros((n, EDGE_DIM), dtype=torch.float32)
        return Data(x=torch.tensor(nf), edge_index=ei, edge_attr=ea,
                    y=torch.tensor(ylabel))

    src, dst, edge_feat = [], [], []
    for (s, d), st in pair_stats.items():
        cnt = st["cnt"]
        # Edge features (d_e = 10):
        # [pkt_count, len_mean, interval_mean,
        #  rdao_mean, rdio_mean, sdio_mean, sdao_mean,
        #  dio_ratio, dao_ratio, is_reverse]
        feat = [
            cnt, st["len"] / cnt, st["intv"] / cnt,
            st["rdao"] / cnt, st["rdio"] / cnt, st["sdio"] / cnt, st["sdao"] / cnt,
            st["dio"] / cnt, st["dao"] / cnt, 0.0,
        ]
        rev = feat[:-1] + [1.0]
        src.append(s); dst.append(d); edge_feat.append(feat)
        src.append(d); dst.append(s); edge_feat.append(rev)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat, dtype=torch.float32)

    return Data(x=torch.tensor(nf), edge_index=edge_index,
                edge_attr=edge_attr, y=torch.tensor(ylabel))


def load_uos_csv(csv_dir: str, window_seconds: float = 10.0):
    """Đọc mọi CSV UOS trong thư mục, gộp thành list[Data].

    Mỗi FILE là một kịch bản (scenario) độc lập -> windows của các file
    không trộn thời gian với nhau (mỗi file build riêng rồi gộp danh sách).

    Chống chịu: bỏ qua file rỗng / lỗi schema, vẫn tiếp tục các file còn lại.
    """
    # Bộ UOS_IOTSH_2024 có cấu trúc thư mục BA TẦNG:
    #   1-Normal_Traffic / 2-Single_Attacker / 3-Dual_Attackers
    #     └── 1-Small_Network / 2-Medium_Network_Single_DODAG / 3-..._Dual_DODAGs
    #           └── *.csv
    # Bản trước chỉ quét thư mục phẳng nên trả về 0 đồ thị. Quét ĐỆ QUY.
    files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not files:
        log.error("Không tìm thấy CSV nào dưới %s", csv_dir)
        return []
    log.info("UOS: tìm thấy %d tệp CSV | cửa sổ = %.1f giây", len(files), window_seconds)
    all_graphs = []
    n_ok, n_skip = 0, 0
    for f in files:
        try:
            if os.path.getsize(f) < 16:          # file rỗng / chỉ header
                log.warning("Bỏ qua (rỗng): %s", os.path.basename(f))
                n_skip += 1
                continue
            df = pd.read_csv(f)
            if df.shape[0] == 0:
                log.warning("Bỏ qua (0 dòng): %s", os.path.basename(f))
                n_skip += 1
                continue
            has_label = "label" in [c.strip() for c in df.columns]
            gs = build_graphs_from_df(df, window_seconds)
            # Ghi nhận nguồn gốc: tên tệp và nhóm kịch bản (thư mục cấp 1).
            # Cho phép chia tập THEO KỊCH BẢN thay vì ngẫu nhiên — nghiêm ngặt
            # hơn nhiều vì mỗi tệp là một lần mô phỏng độc lập với vị trí kẻ
            # tấn công khác nhau.
            rel = os.path.relpath(f, csv_dir).replace("\\", "/")
            scen = rel.split("/")[0] if "/" in rel else "root"
            fid = os.path.splitext(os.path.basename(f))[0]
            for g in gs:
                g.src_file = fid
                g.scenario = scen
            all_graphs.extend(gs)
            n_ok += 1
            log.info("UOS %s [label=%s] -> %d graph",
                     os.path.basename(f), "CÓ" if has_label else "KHÔNG→Normal", len(gs))
        except Exception as e:                       # noqa: BLE001
            log.warning("Bỏ qua %s: %s", os.path.basename(f), e)
            n_skip += 1
    log.info("UOS tổng: %d file OK, %d file bỏ qua, %d graph",
             n_ok, n_skip, len(all_graphs))
    return all_graphs


def split_uos_by_scenario(graphs, train_scen, val_scen, test_scen, key="scenario"):
    """Chia tập theo KỊCH BẢN hoặc theo TỆP, thay vì ngẫu nhiên.

    key="scenario": chia theo nhóm thư mục cấp 1 (1-Normal_Traffic,
        2-Single_Attacker, 3-Dual_Attackers).
    key="src_file": chia theo từng tệp mô phỏng. NGHIÊM NGẶT NHẤT với bộ UOS,
        vì mỗi tệp là một lần mô phỏng với VỊ TRÍ KẺ TẤN CÔNG KHÁC NHAU
        (tên tệp mã hoá ID node tấn công, ví dụ ..._SingleDODAG_10.csv là
        node 10 tấn công). Chia theo tệp buộc mô hình khái quát sang kẻ tấn
        công ở vị trí chưa từng thấy — đúng yêu cầu triển khai thực tế, và
        khắc phục được đúng điểm yếu của bộ IoT-RPL 2021 (node tấn công cố định).
    """
    def norm(x):
        """Tách danh sách; chấp nhận cả ',' và '+' làm dấu phân cách.

        Dấu '+' hữu ích khi cần GHÉP nhiều nhóm vào một tập, ví dụ ghép
        1-Normal_Traffic vào cùng tập với 3-Dual_Attackers để tập đó có đủ
        hai lớp. Nhóm 1-Normal_Traffic chỉ chứa lớp Normal nên KHÔNG BAO GIỜ
        được đứng một mình làm tập kiểm định hay kiểm thử.
        """
        if isinstance(x, str):
            x = x.replace("+", ",").split(",")
        return {str(t).strip() for t in x if str(t).strip()}

    tr_s, va_s, te_s = norm(train_scen), norm(val_scen), norm(test_scen)
    ov = (tr_s & te_s) | (tr_s & va_s) | (va_s & te_s)
    if ov:
        raise ValueError("Trùng giữa các tập: %s" % sorted(ov))

    train, val, test, unk = [], [], [], set()
    for g in graphs:
        k = str(getattr(g, key, "")).strip()
        if k in tr_s:
            train.append(g)
        elif k in va_s:
            val.append(g)
        elif k in te_s:
            test.append(g)
        else:
            unk.add(k)
    if unk:
        log.warning("%d giá trị '%s' không thuộc tập nào: %s",
                    len(unk), key, sorted(unk)[:6])
    log.info("UOS chia theo %s — train=%d val=%d test=%d đồ thị",
             key, len(train), len(val), len(test))

    # ── KIỂM TRA BẮT BUỘC: mỗi tập phải có ĐỦ HAI LỚP ──
    # Nếu tập kiểm định chỉ chứa một lớp, macro-F1 kiểm định trở nên vô nghĩa
    # và cơ chế chọn điểm kiểm tra sẽ hỏng hoàn toàn — mô hình "tốt nhất" được
    # chọn theo một chỉ số không phân biệt được gì.
    import torch as _t
    for nm, gs in (("train", train), ("val", val), ("test", test)):
        if not gs:
            log.error("Tập %s RỖNG — kiểm tra lại danh sách %s.", nm, key)
            continue
        cls = sorted({int(v) for g in gs for v in _t.unique(g.y)})
        if len(cls) < 2:
            log.error("Tập %s CHỈ CÓ LỚP %s. macro-F1 sẽ vô nghĩa. "
                      "Cần đưa cả tệp Normal LẪN tệp tấn công vào mỗi tập.",
                      nm, cls)
    return train, val, test


def list_uos_sources(graphs):
    """In danh sách kịch bản và tệp có trong tập đồ thị — dùng để soạn lệnh chia."""
    scen, files = {}, {}
    for g in graphs:
        scen[str(getattr(g, "scenario", "?"))] = scen.get(str(getattr(g, "scenario", "?")), 0) + 1
        files[str(getattr(g, "src_file", "?"))] = files.get(str(getattr(g, "src_file", "?")), 0) + 1
    log.info("Kịch bản: %s", scen)
    log.info("Số tệp: %d", len(files))
    return scen, files
