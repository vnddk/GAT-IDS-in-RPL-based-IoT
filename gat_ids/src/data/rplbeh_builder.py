"""
rplbeh_builder.py
=================
Chuyen bo du lieu RPL-IDS-Beh (Nassrullah & Alisa, IJIES 2025) tu dang bang
phang (da bi xao tron) ve dang DO THI theo cua so thoi gian, tuong thich voi
pipeline EdgeGAT-IDS (cung giao dien voi radar_builder).

Ba buoc:
  1. UNSHUFFLE  - dao nguoc phep xao tron xac dinh cua tac gia
                  (pandas .sample(frac=1, random_state=42)) de lay lai thu tu
                  ghi goc: [file log] -> [interval tang dan] -> [node_id tang dan]
  2. SEGMENT    - cat thu tu goc thanh cac mo phong rieng biet (sim_id)
  3. BUILD      - moi cap (sim_id, time_win) -> mot do thi DODAG:
                  node = ban ghi node, canh = quan he cha-con theo parent_id

Chay doc lap:
    python rplbeh_builder.py --csv RPL-IDS-Beh.csv --out data/rplbeh --audit

Sinh ra:
    data/rplbeh/graphs.npz   (khong can torch)
    data/rplbeh/meta.json
va neu co torch_geometric:
    data/rplbeh/{train,val,test}.pt   (list[Data])
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Dinh nghia dac trung
# --------------------------------------------------------------------------- #

ID_COLS = ["time_sec", "node_id", "parent_id"]          # metadata, KHONG dua vao x
DEAD_COLS = ["nbr_dao_ack_rcv"]                          # hang so 0 tren toan bo CSV
LABEL_COL = "label"

CLASS_NAMES = ["Normal", "VNA", "DRA", "DISA", "SFA"]

# 24 dac trung node "sach" (loai ID + cot chet)
NODE_FEATS = [
    "rpl_ver", "rpl_rank", "dis_sent", "dio_sent", "dao_sent",
    "nbr_dis_rcv", "nbr_dio_rcv", "nbr_fwd_to_me", "nbr_fwd_to_others",
    "nbr_fwd_bcast", "nbr_rpl_ctrl", "nbr_non_rpl_ctrl", "nbr_rpl_ver_rcv",
    "nbr_rpl_rank_rcv", "nbr_fwd_rpl", "nbr_fwd_non_rpl",
    "diff_rpl_rank", "diff_rpl_ver", "norm_rank_diff", "ctrl_to_data_ratio",
    "non_rpl_to_rpl_ratio", "rpl_fwd_ratio", "non_rpl_fwd_ratio",
    "total_fwd_ratio",
]

EDGE_FEAT_NAMES = [
    "abs_rank_diff",      # |rank_u - rank_v|
    "signed_rank_gap",    # rank_child - rank_parent  (< 0  => dau hieu DRA)
    "abs_ver_diff",       # |ver_u - ver_v|           (dau hieu VNA)
    "ctrl_vol_parent",    # dis+dio+dao cua dau tren
    "ctrl_vol_child",     # dis+dio+dao cua dau duoi
    "fwd_to_me_child",    # nbr_fwd_to_me cua con     (dau hieu SFA)
    "fwd_ratio_gap",      # total_fwd_ratio(child) - total_fwd_ratio(parent)
    "dis_rate_gap",       # dis_sent(child) - dis_sent(parent)  (dau hieu DISA)
    "n_siblings",         # so con cua node cha (bac cua cha)
    "direction",          # 1 = con -> cha (up), 0 = cha -> con (down)
]


@dataclass
class BuildConfig:
    csv_path: str
    out_dir: str = "data/rplbeh"
    shuffle_seed: int = 42          # random_state ma tac gia dung
    add_self_loops: bool = True
    bidirectional: bool = True      # them ca canh cha->con
    connect_orphans: bool = True    # node mat cha -> noi vao node co rank nho nhat
    min_graph_size: int = 4         # bo do thi qua nho
    log1p: bool = True              # nen dai dong cua cac dac trung dem
    split_by: str = "sim"           # "sim" (khuyen nghi) | "window" | "row"
    split_ratio: tuple = (0.7, 0.15, 0.15)
    seed: int = 2470165
    node_feats: list = field(default_factory=lambda: list(NODE_FEATS))


# --------------------------------------------------------------------------- #
# Buoc 1 - dao nguoc xao tron
# --------------------------------------------------------------------------- #

def unshuffle(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Dao nguoc `df.sample(frac=1, random_state=seed)` trong dataset_generation.py.

    pandas dung numpy RandomState -> hoan vi tai lap duoc chinh xac.
    Tra ve DataFrame theo dung thu tu ma generate_csv() da ghi ra.
    """
    n = len(df)
    perm = np.random.RandomState(seed).permutation(n)
    order = np.argsort(perm)
    out = df.iloc[order].reset_index(drop=True)

    # Kiem chung: trong cung mot interval, node_id phai TANG NGHIEM NGAT
    t, nd = out["time_sec"].values, out["node_id"].values
    same = np.diff(t) == 0
    if same.sum() and (np.diff(nd)[same] > 0).mean() < 0.999:
        raise RuntimeError(
            "Dao xao tron THAT BAI: node_id khong tang trong cung interval. "
            "Co the CSV da bi xao tron lai hoac sinh boi phien ban script khac."
        )
    return out


# --------------------------------------------------------------------------- #
# Buoc 2 - cat thanh tung mo phong
# --------------------------------------------------------------------------- #

def segment_simulations(df: pd.DataFrame) -> pd.DataFrame:
    """Gan sim_id. generate_csv() ghi lan luot tung file log; trong moi file
    interval tang dan va node_id tang dan -> ranh gioi file la cho vi pham."""
    t, nd = df["time_sec"].values, df["node_id"].values
    t0 = t.min()
    brk = np.r_[
        True,
        ((t[1:] == t0) & (t[:-1] != t0))          # bat dau interval dau tien
        | (t[1:] < t[:-1])                        # thoi gian lui
        | ((t[1:] == t[:-1]) & (nd[1:] <= nd[:-1]))  # node_id khong tang
    ]
    df = df.copy()
    df["sim_id"] = np.cumsum(brk) - 1
    return df


# --------------------------------------------------------------------------- #
# Buoc 3 - dung do thi
# --------------------------------------------------------------------------- #



def _build_one(sub: pd.DataFrame, cfg: BuildConfig):
    """Dung mot do thi tu cac ban ghi cua mot (sim_id, time_sec)."""
    sub = sub.sort_values("node_id").reset_index(drop=True)
    ids = sub["node_id"].values
    pos = {int(v): i for i, v in enumerate(ids)}

    x = sub[cfg.node_feats].to_numpy(dtype=np.float32)
    if cfg.log1p:
        # chi nen cac cot dem (khong nen ty le/hieu da chuan hoa)
        ratio_like = {"norm_rank_diff", "rpl_fwd_ratio",
                      "non_rpl_fwd_ratio", "total_fwd_ratio"}
        for j, name in enumerate(cfg.node_feats):
            if name not in ratio_like:
                x[:, j] = np.log1p(np.clip(x[:, j], 0, None))

    y = sub[LABEL_COL].to_numpy(dtype=np.int64)

    # --- canh cha-con ---
    pairs = []  # (child_local, parent_local)
    for i, p in enumerate(sub["parent_id"].values):
        p = int(p)
        if p in pos and pos[p] != i:
            pairs.append((i, pos[p]))

    if cfg.connect_orphans and pairs:
        root_local = int(np.argmin(sub["rpl_rank"].replace(0, np.inf).values))
        linked = {c for c, _ in pairs}
        for i in range(len(sub)):
            if i not in linked and i != root_local:
                pairs.append((i, root_local))

    if not pairs:
        return None

    child_deg = {}
    for _, p in pairs:
        child_deg[p] = child_deg.get(p, 0) + 1

    rank = sub["rpl_rank"].to_numpy(np.float32)
    ver = sub["rpl_ver"].to_numpy(np.float32)
    ctrl = sub[["dis_sent", "dio_sent", "dao_sent"]].to_numpy(np.float32).sum(1)
    fwd_me = sub["nbr_fwd_to_me"].to_numpy(np.float32)
    fwd_ratio = sub["total_fwd_ratio"].to_numpy(np.float32)
    dis = sub["dis_sent"].to_numpy(np.float32)

    src, dst, eattr = [], [], []
    for c, p in pairs:
        feat = [
            abs(rank[p] - rank[c]),
            rank[c] - rank[p],
            abs(ver[p] - ver[c]),
            np.log1p(ctrl[p]),
            np.log1p(ctrl[c]),
            np.log1p(fwd_me[c]),
            fwd_ratio[c] - fwd_ratio[p],
            np.log1p(max(dis[c] - dis[p], 0.0)),
            float(child_deg.get(p, 0)),
            1.0,  # huong: con -> cha
        ]
        src.append(c); dst.append(p); eattr.append(feat)
        if cfg.bidirectional:
            f2 = list(feat); f2[-1] = 0.0
            src.append(p); dst.append(c); eattr.append(f2)

    if cfg.add_self_loops:
        for i in range(len(sub)):
            src.append(i); dst.append(i)
            eattr.append([0.0] * (len(EDGE_FEAT_NAMES) - 1) + [0.5])

    return dict(
        x=x,
        y=y,
        edge_index=np.asarray([src, dst], dtype=np.int64),
        edge_attr=np.asarray(eattr, dtype=np.float32),
        node_ids=ids.astype(np.int64),
        sim_id=int(sub["sim_id"].iloc[0]),
        time_win=int(sub["time_sec"].iloc[0]),
    )


def build_graphs(df: pd.DataFrame, cfg: BuildConfig):
    graphs = []
    for (_sid, _tw), sub in df.groupby(["sim_id", "time_sec"], sort=True):
        if len(sub) < cfg.min_graph_size:
            continue
        g = _build_one(sub, cfg)
        if g is not None:
            graphs.append(g)
    return graphs


# --------------------------------------------------------------------------- #
# Chia tap - MAC DINH THEO MO PHONG (chong ro ri thoi gian)
# --------------------------------------------------------------------------- #

def split_graphs(graphs, cfg: BuildConfig):
    rng = np.random.default_rng(cfg.seed)
    if cfg.split_by == "sim":
        keys = np.array(sorted({g["sim_id"] for g in graphs}))
    elif cfg.split_by == "window":
        keys = np.arange(len(graphs))
    else:
        raise ValueError("split_by phai la 'sim' hoac 'window'")

    rng.shuffle(keys)
    n = len(keys)
    a = int(cfg.split_ratio[0] * n)
    b = a + int(cfg.split_ratio[1] * n)
    tr, va, te = set(keys[:a]), set(keys[a:b]), set(keys[b:])

    if cfg.split_by == "sim":
        pick = lambda S: [g for g in graphs if g["sim_id"] in S]
    else:
        pick = lambda S: [g for i, g in enumerate(graphs) if i in S]
    return pick(tr), pick(va), pick(te)


# --------------------------------------------------------------------------- #
# Kiem toan ro ri (phien ban rut gon cua feature_audit.py)
# --------------------------------------------------------------------------- #

def audit(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    from sklearn.metrics import mutual_info_score

    y = df[LABEL_COL].values
    Hy = -sum(p * np.log(p) for p in np.bincount(y)[np.bincount(y) > 0] / len(y))
    Hb = -sum(p * np.log(p) for p in
              [np.mean(y > 0), np.mean(y == 0)] if p > 0)
    ceil_bin = Hb / Hy  # tran nMI cua mot oracle nhi phan hoan hao

    rows = []
    for c in feats:
        b = pd.qcut(df[c], 32, duplicates="drop", labels=False)
        nmi = mutual_info_score(b.fillna(-1), y) / Hy
        rows.append((c, nmi, nmi / ceil_bin))
    out = (pd.DataFrame(rows, columns=["feature", "nMI", "frac_of_binary_ceiling"])
             .sort_values("nMI", ascending=False))
    out.attrs["ceiling"] = ceil_bin
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/rplbeh")
    ap.add_argument("--split-by", default="sim", choices=["sim", "window"])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--no-orphan-link", action="store_true")
    a = ap.parse_args()

    cfg = BuildConfig(csv_path=a.csv, out_dir=a.out, split_by=a.split_by,
                      connect_orphans=not a.no_orphan_link)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    print(f"[1/4] Doc CSV: {df.shape}")

    df = unshuffle(df, cfg.shuffle_seed)
    print("[2/4] Dao xao tron: OK (node_id tang don dieu trong moi interval)")

    df = segment_simulations(df)
    n_sim = df["sim_id"].nunique()
    print(f"[3/4] Cat mo phong: {n_sim} mo phong "
          f"(bai bao cong bo 96 - chenh lech can ghi chu trong luan van)")

    if a.audit:
        rep = audit(df, cfg.node_feats + ID_COLS)
        print(f"\n--- KIEM TOAN RO RI (tran oracle nhi phan = "
              f"{rep.attrs['ceiling']:.3f}) ---")
        print(rep.head(12).to_string(index=False))
        flag = rep[rep.frac_of_binary_ceiling > 0.6]
        print("CANH BAO ro ri:" if len(flag) else
              "Khong phat hien ro ri nhan (khac han RADAR).")
        rep.to_csv(os.path.join(cfg.out_dir, "feature_audit.csv"), index=False)

    graphs = build_graphs(df, cfg)
    tr, va, te = split_graphs(graphs, cfg)
    sizes = [g["x"].shape[0] for g in graphs]
    print(f"\n[4/4] Do thi: {len(graphs)} | node TB {np.mean(sizes):.1f} "
          f"(min {min(sizes)}, max {max(sizes)})")
    print(f"      Chia theo '{cfg.split_by}': "
          f"train {len(tr)} / val {len(va)} / test {len(te)}")

    np.savez_compressed(
        os.path.join(cfg.out_dir, "graphs.npz"),
        **{f"{k}_{i}": g[k]
           for i, g in enumerate(graphs)
           for k in ("x", "y", "edge_index", "edge_attr", "node_ids")},
    )
    meta = dict(
        n_graphs=len(graphs), n_sim=int(n_sim),
        d_node=len(cfg.node_feats), d_edge=len(EDGE_FEAT_NAMES),
        node_feats=cfg.node_feats, edge_feats=EDGE_FEAT_NAMES,
        class_names=CLASS_NAMES, num_classes=5,
        split_by=cfg.split_by,
        split_index=dict(
            train=[g["sim_id"] for g in tr][:0] or sorted({g["sim_id"] for g in tr}),
            val=sorted({g["sim_id"] for g in va}),
            test=sorted({g["sim_id"] for g in te}),
        ),
    )
    with open(os.path.join(cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Xuat sang PyG neu co
    try:
        import torch
        from torch_geometric.data import Data

        def to_pyg(gs):
            return [Data(x=torch.tensor(g["x"]),
                         edge_index=torch.tensor(g["edge_index"]),
                         edge_attr=torch.tensor(g["edge_attr"]),
                         y=torch.tensor(g["y"]),
                         node_ids=torch.tensor(g["node_ids"]),
                         sim_id=g["sim_id"], time_win=g["time_win"])
                    for g in gs]

        for name, gs in [("train", tr), ("val", va), ("test", te)]:
            torch.save(to_pyg(gs), os.path.join(cfg.out_dir, f"{name}.pt"))
        print("      Da xuat train/val/test.pt (PyG)")
    except ImportError:
        print("      (Khong tim thay torch_geometric - chi luu graphs.npz)")


if __name__ == "__main__":
    main()


# =========================================================================== #
#  Giao dien cho framework — chuyen dict -> PyG Data, gan metadata thoi gian
# =========================================================================== #

from ..utils.common import get_logger
log = get_logger()

RPLBEH_CLASSES = ["Normal", "VNA", "DRA", "DISA", "SFA"]


def load_rplbeh(csv_dir: str, window_seconds: float = None, **kw):
    """Doc RPL-IDS-Beh.csv -> list[torch_geometric.data.Data].

    QUY TRINH KHOI PHUC THU TU THOI GIAN (co bang chung kiem chung):

      1. `dataset_generation.py` cua tac gia ket thuc bang
         `df.sample(frac=1, random_state=42)`. pandas dung numpy RandomState
         nen hoan vi TAI LAP DUOC chinh xac -> `unshuffle()` dao nguoc no.

      2. KIEM CHUNG BAT BUOC: sau khi dao, node_id phai TANG NGHIEM NGAT
         trong moi interval — dung thu tu ma `generate_csv()` ghi ra.
         Do duoc tren du lieu that: 151.224/151.224 cap lien tiep thoa,
         ti le 1,000000, KHONG mot vi pham nao. Truoc khi dao ti le nay
         chi la 0,4951 (tuc ngau nhien). `unshuffle()` nem RuntimeError
         neu ti le tut duoi 0,999.

      3. `segment_simulations()` cat lai 95 lan chay (bai bao cong bo 96 —
         mot tep log 10-node duong nhu khong sinh dong nao; can ghi chu).

      4. KIEM CHUNG TINH LIEN TUC: trong ca 95/95 mo phong, cac cua so cach
         deu dung 8 don vi, tu 8 toi 592, 74 cua so moi mo phong.
         `time_sec` la chi so PHUT (`interval = timestamp // 60_000_000`),
         nen 8 phut = 480 s = dung "Capture Interval 480 seconds" o Bang 1
         cua bai bao. Day la mot MIEN THOI GIAN LIEN TUC dung nghia.

    Vi (4) nen bo du lieu nay DUNG DUOC cho ATA: moi mo phong la mot chuoi
    thoi gian deu dan, khong dut quang.

    `window_seconds` bi BO QUA: do phan giai cua so do bo du lieu an dinh o
    480 s va khong tai phan duoc tu CSV.
    """
    import glob
    import torch
    from torch_geometric.data import Data

    if window_seconds:
        log.info("RPL-Beh: bo qua window_seconds=%.1f — cua so co dinh 480 s "
                 "do bo du lieu an dinh.", window_seconds)

    if os.path.isfile(csv_dir):
        files = [csv_dir]
    else:
        files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not files:
        log.error("Khong tim thay CSV trong %s", csv_dir)
        return []
    if len(files) > 1:
        log.warning("Tim thay %d tep; chi dung tep dau: %s", len(files), files[0])

    cfg = BuildConfig(csv_path=files[0])
    df = pd.read_csv(files[0])
    log.info("RPL-Beh: %d dong x %d cot", *df.shape)

    df = segment_simulations(unshuffle(df, cfg.shuffle_seed))
    raw = build_graphs(df, cfg)

    out = []
    for g in raw:
        d = Data(x=torch.tensor(g["x"]),
                 edge_index=torch.tensor(g["edge_index"]),
                 edge_attr=torch.tensor(g["edge_attr"]),
                 y=torch.tensor(g["y"]))
        # Metadata thoi gian — BAT BUOC cho `--split chrono` va cho viec dung
        # mien thoi gian cua ATA. Moi mo phong la mot "tep" doc lap voi dong
        # ho rieng; chi so cua so chinh la thoi gian trong lan chay do.
        d.src_file = f"sim{int(g['sim_id']):03d}"
        d.win_idx = int(g["time_win"])
        d.sim_id = int(g["sim_id"])
        out.append(d)

    sizes = [d.num_nodes for d in out]
    log.info("RPL-Beh: %d do thi | node/do thi TB %.1f (min %d, max %d) | "
             "d_n=%d d_e=%d", len(out), float(np.mean(sizes)), min(sizes),
             max(sizes), out[0].x.shape[1], out[0].edge_attr.shape[1])
    return out
