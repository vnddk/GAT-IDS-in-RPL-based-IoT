"""Loader linh hoạt: ưu tiên CSV thật, fallback synthetic.

Pipeline chuẩn:
  1. Lấy danh sách graph (CSV hoặc synthetic).
  2. Chia train/val/test ở mức GRAPH (tránh rò rỉ thông tin giữa các tập).
  3. Fit scaler CHỈ trên train, áp dụng cho val/test (tránh data leakage).
  4. Trả về 3 DataLoader của PyG.
"""
from __future__ import annotations
import os
import os
import glob
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .synthetic import generate_dataset, CLASS_NAMES
from .uos_builder import load_uos_csv, UOS_CLASSES
from .radar_builder import load_radar, RADAR_CLASSES
from ..utils.common import get_logger

log = get_logger()


from .iotrpl_builder import load_iotrpl, IOTRPL_CLASSES
from .rplbeh_builder import load_rplbeh, RPLBEH_CLASSES

DATASET_CLASSES = {
    "uos": UOS_CLASSES,
    "radar": RADAR_CLASSES,
    "iotrpl": IOTRPL_CLASSES,
    "rplbeh": RPLBEH_CLASSES,
}

def load_from_csv(csv_dir: str, dataset: str = "uos", window_seconds: float = 10.0):
    """Đọc CSV thật thành list[Data].

    Khi THÊM DATASET MỚI:
      1. Viết builder mới trong src/data/ (tham khảo radar_builder.py)
      2. Thêm 1 dòng import + 1 nhánh if ở đây
      3. Thêm tên lớp vào DATASET_CLASSES
    """
    if dataset == "uos":
        return load_uos_csv(csv_dir, window_seconds)
    if dataset == "radar":
        return load_radar(csv_dir, window_seconds)
    if dataset == "rplbeh":
        # Cửa sổ CỐ ĐỊNH 480 s do bộ dữ liệu ấn định (interval = ts // 60e6, bội
        # của 8 phút), không tái phân được từ CSV -> window_seconds bị bỏ qua.
        return load_rplbeh(csv_dir, window_seconds)
    if dataset == "iotrpl":
        # Bộ IoT-RPL KHÔNG có cột thời gian -> window_seconds được diễn giải
        # thành SỐ DÒNG mỗi cửa sổ (mặc định 2000 gói/đồ thị).
        wr = int(window_seconds) if window_seconds and window_seconds > 50 else 2000
        return load_iotrpl(csv_dir, window_rows=wr,
                           skip_head=int(os.environ.get("IOTRPL_SKIP_HEAD", 300000)))
    log.warning("Chưa có builder cho dataset=%s, trả rỗng.", dataset)
    return []


# ──────────────────────────────────────────────────────────────
#  Chia tập theo graph, stratified theo nhãn trội của graph
# ──────────────────────────────────────────────────────────────
def _graph_label(data):
    """Nhãn đại diện của 1 graph để phân tầng.

    SỬA LỖI: bản gốc lấy lớp ĐA SỐ, nhưng graph RADAR ~16 node chỉ có 1-3 node
    tấn công nên lớp đa số LUÔN là Normal(0) -> mọi graph cùng một rổ -> phân
    tầng vô hiệu, lớp hiếm (version) có thể biến mất khỏi test. Bản mới lấy lớp
    tấn công HIẾM NHẤT trong graph (nếu có), để mỗi lớp có rổ riêng.
    """
    vals, counts = torch.unique(data.y, return_counts=True)
    atk = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != 0]
    if not atk:
        return 0
    return min(atk, key=lambda t: t[1])[0]


def _node_class_counts(graphs):
    cnt = {}
    for g in graphs:
        for v, c in zip(*torch.unique(g.y, return_counts=True)):
            cnt[int(v)] = cnt.get(int(v), 0) + int(c)
    return cnt


def split_graphs(graphs, ratios=(0.7, 0.15, 0.15), seed=42, num_classes=None):
    rng = np.random.default_rng(seed)
    by_label = {}
    for g in graphs:
        by_label.setdefault(_graph_label(g), []).append(g)

    train, val, test = [], [], []
    for label, items in by_label.items():
        idx = rng.permutation(len(items))
        n = len(items)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        for j, i in enumerate(idx):
            if j < n_tr:
                train.append(items[i])
            elif j < n_tr + n_va:
                val.append(items[i])
            else:
                test.append(items[i])

    # Bảo đảm mọi lớp có node ở TEST (nếu không, macro-F1 tính thiếu lớp).
    c_te = _node_class_counts(test)
    all_cls = set(_node_class_counts(train)) | set(c_te) | set(_node_class_counts(val))
    if num_classes:
        all_cls |= set(range(num_classes))
    for c in sorted(all_cls):
        if c == 0 or c_te.get(c, 0) > 0:
            continue
        donors = [i for i, g in enumerate(train) if int((g.y == c).sum()) > 0]
        if not donors:
            log.warning("Lớp %d không có node ở tập nào — kiểm tra gán nhãn.", c)
            continue
        n_move = max(1, int(0.15 * len(donors)))
        for i in sorted(donors[:n_move], reverse=True):
            test.append(train.pop(i))
        log.warning("Lớp %d vắng ở test -> chuyển %d graph train->test.", c, n_move)

    # In phân bố node để soi trực tiếp
    ctr, cva, cte = (_node_class_counts(x) for x in (train, val, test))
    log.info("── PHÂN BỐ NODE THEO LỚP (train/val/test) ──")
    for c in sorted(all_cls):
        log.info("   lớp %-3d %8d / %7d / %7d", c, ctr.get(c,0), cva.get(c,0), cte.get(c,0))

    rng.shuffle(train)
    return train, val, test


# ──────────────────────────────────────────────────────────────
#  Chuẩn hóa (fit trên train, transform tất cả) — chống data leakage
# ──────────────────────────────────────────────────────────────
class GraphScaler:
    """Z-score scaler riêng cho node features và edge features."""
    def __init__(self):
        self.x_mean = self.x_std = None
        self.e_mean = self.e_std = None

    def fit(self, graphs):
        xs = torch.cat([g.x for g in graphs], dim=0)
        es = torch.cat([g.edge_attr for g in graphs], dim=0)
        self.x_mean, self.x_std = xs.mean(0), xs.std(0).clamp_min(1e-6)
        self.e_mean, self.e_std = es.mean(0), es.std(0).clamp_min(1e-6)
        return self

    def transform(self, graphs):
        for g in graphs:
            g.x = (g.x - self.x_mean) / self.x_std
            g.edge_attr = (g.edge_attr - self.e_mean) / self.e_std
        return graphs


# ──────────────────────────────────────────────────────────────
#  API chính
# ──────────────────────────────────────────────────────────────
def get_datasets(cfg: dict):
    """Trả về (train_graphs, val_graphs, test_graphs, meta)."""
    dcfg = cfg["data"]
    source = dcfg["source"]

    graphs = []
    if source in ("csv", "auto"):
        dataset = dcfg.get("dataset", "uos")
        graphs = load_from_csv(dcfg["csv_dir"], dataset, dcfg.get("window_seconds", 10))

    used_csv = bool(graphs)
    if not graphs:                       # synthetic hoặc fallback
        if source == "csv":
            raise FileNotFoundError(
                f"source=csv nhưng không đọc được CSV ở {dcfg['csv_dir']}"
            )
        s = dcfg["synthetic"]
        log.info("Sinh synthetic: %d graph", s["n_graphs"])
        graphs = generate_dataset(
            n_graphs=s["n_graphs"],
            nodes_range=tuple(s["nodes_per_graph"]),
            seed=cfg["seed"],
        )

    # Chọn danh sách lớp theo dataset
    if used_csv:
        ds_name = dcfg.get("dataset", "uos")
        class_names = DATASET_CLASSES.get(ds_name, dcfg["classes"])
        # Nếu RADAR nhưng thực tế chỉ có vài lớp, giới hạn theo dữ liệu thật
        actual_labels = set()
        for g in graphs:
            actual_labels.update(g.y.unique().tolist())
        # Giữ nguyên danh sách gốc nhưng log cảnh báo nếu thiếu lớp
        if len(actual_labels) < len(class_names):
            log.info("Dữ liệu chỉ có %d/%d lớp: %s",
                     len(actual_labels), len(class_names), sorted(actual_labels))
    else:
        class_names = dcfg["classes"]

    train, val, test = split_graphs(graphs, tuple(dcfg["split"]), cfg["seed"],
                                    num_classes=len(class_names))

    # Chuẩn hoá: mặc định Yeo-Johnson (ổn định phương sai). Tắt: data.use_power=false
    if dcfg.get("use_power", True):
        from .transforms import GraphPowerScaler
        scaler = GraphPowerScaler(use_power=True).fit(train)
    else:
        scaler = GraphScaler().fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    meta = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "n_node_features": train[0].x.shape[1],
        "n_edge_features": train[0].edge_attr.shape[1],
        "scaler": scaler,
    }
    log.info("Split: train=%d val=%d test=%d | d_n=%d d_e=%d",
             len(train), len(val), len(test),
             meta["n_node_features"], meta["n_edge_features"])
    return train, val, test, meta


def oversample_rare_classes(train, num_classes, min_ratio=0.02, max_factor=20,
                             mode="balance"):
    """Oversample lớp hiếm. Hai chế độ:

    mode="balance" (KHUYẾN NGHỊ — mặc định):
        Đưa MỌI lớp tấn công (1..C-1) về cùng mức node của lớp tấn công LỚN NHẤT.
        factor_c = clip(target_nodes / nodes_c, 1, max_factor), hỗ trợ hệ số lẻ
        (1.4x = nhân bản 40% số graph chứa lớp đó, chọn ngẫu nhiên).
        -> Bảo toàn trật tự tương đối giữa các lớp tấn công: không lớp nào bị
        "pha loãng" so với lớp khác. Normal không đổi (đã dư thừa).

        Bài học từ thực nghiệm v1: chế độ ngưỡng (ratio) nhân x10-x20 các lớp
        siêu hiếm nhưng BỎ QUA lớp trung bình ngay dưới ngưỡng (rank 1.7%,
        hệ số 1.18 -> làm tròn 1) khiến tỉ trọng tương đối của chúng rơi từ
        1.7% xuống 0.5% trong phân phối mới -> rank F1 sụp 0.676 -> 0.199.

    mode="ratio" (legacy, dùng ở v1):
        Lớp nào dưới min_ratio tổng node -> nhân nguyên bản đến khi đạt min_ratio.

    Không sinh dữ liệu giả (khác SMOTE), không đổi loss (khác class weight).
    """
    import random
    counts = torch.zeros(num_classes)
    for g in train:
        for c in g.y:
            counts[int(c)] += 1
    total = counts.sum().item()

    # graphs chứa từng lớp (RADAR: mỗi graph thường chỉ chứa Normal + 1 lớp attack)
    graphs_of = {c: [] for c in range(num_classes)}
    for g in train:
        for c in set(int(v) for v in g.y.unique()):
            graphs_of[c].append(g)

    extra = []
    factors_log = {}

    if mode == "balance":
        attack_counts = counts[1:]
        if attack_counts.max() == 0:
            log.info("Oversample(balance): không có lớp tấn công -> giữ nguyên")
            return train
        target = attack_counts.max().item()
        for c in range(1, num_classes):
            n_c = counts[c].item()
            if n_c == 0 or not graphs_of[c]:
                continue
            f = min(target / n_c, float(max_factor))
            if f <= 1.01:
                continue
            gl = graphs_of[c]
            n_extra = int(round((f - 1.0) * len(gl)))
            if n_extra > 0:
                extra.extend(random.choices(gl, k=n_extra))
                factors_log[c] = f"x{f:.1f}"
    else:  # mode == "ratio" (legacy)
        for c in range(num_classes):
            n_c = counts[c].item()
            if n_c == 0:
                continue
            if n_c / total < min_ratio:
                f = min(int(min_ratio * total / n_c), max_factor)
                if f > 1 and graphs_of[c]:
                    gl = graphs_of[c]
                    extra.extend(gl * (f - 1))
                    factors_log[c] = f"x{f}"

    if not extra:
        log.info("Oversample(%s): không cần -> giữ nguyên %d graph", mode, len(train))
        return train

    out = list(train) + extra
    random.shuffle(out)
    log.info("Oversample(%s) %s -> train %d -> %d graph (+%d)",
             mode, factors_log, len(train), len(out), len(extra))
    return out


def smote_augment_graphs(train, num_classes, k=5, max_factor=20, seed=42):
    """SMOTE cho node classification trên đồ thị (GraphSMOTE-lite).

    Khác oversampling (nhân bản NGUYÊN graph -> lặp y hệt, gây overfit và
    pha loãng lớp khác), SMOTE SINH NODE THIỂU SỐ TỔNG HỢP bằng nội suy:
        x_syn = x_i + lambda * (x_nn - x_i),  lambda ~ U(0,1)
    với x_i là node thật lớp c, x_nn là 1 trong k láng giềng gần nhất
    CÙNG LỚP (kNN trong không gian feature, xuyên graph).

    Node tổng hợp được CHÈN VÀO CHÍNH GRAPH CHỦ (graph chứa x_i):
      - copy toàn bộ cạnh của x_i (cả 2 chiều, kèm edge_attr)
      -> node giả nằm trong topology thật, GNN học được ngữ cảnh.

    Graph chủ được THAY THẾ bằng bản augmented (không nhân bản graph gốc)
    -> lớp trung bình giữ nguyên số node tuyệt đối, không bị lặp dữ liệu.

    Mục tiêu: mọi lớp tấn công đạt mức node của lớp tấn công lớn nhất
    (trần max_factor lần số node gốc).
    """
    import random
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Pool node theo lớp: (graph_idx, node_idx) + ma trận feature
    pool_idx = {c: [] for c in range(num_classes)}
    for gi, g in enumerate(train):
        for ni in range(g.y.shape[0]):
            pool_idx[int(g.y[ni])].append((gi, ni))

    counts = {c: len(v) for c, v in pool_idx.items()}
    attack_counts = {c: n for c, n in counts.items() if c > 0 and n > 0}
    if not attack_counts:
        log.info("SMOTE: không có lớp tấn công -> giữ nguyên")
        return train
    target = max(attack_counts.values())

    # Node tổng hợp cần thêm cho từng graph: {graph_idx: [(class, feat_vector, src_node_idx)]}
    additions = {}
    stats = {}
    for c, n_c in attack_counts.items():
        n_syn = min(target, n_c * max_factor) - n_c
        if n_syn <= 0:
            continue
        idx_list = pool_idx[c]
        X = np.stack([train[gi].x[ni].numpy() for gi, ni in idx_list])
        if n_c >= 2:
            kk = min(k, n_c - 1)
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
            _, nbrs = nn.kneighbors(X)          # cột 0 là chính nó
        else:
            nbrs = None                          # 1 node duy nhất -> jitter
        for _ in range(n_syn):
            i = rng.randrange(n_c)
            if nbrs is not None:
                j = int(nbrs[i][rng.randrange(1, nbrs.shape[1])])
                lam = rng.random()
                x_syn = X[i] + lam * (X[j] - X[i])
            else:
                x_syn = X[i] + np_rng.normal(0, 0.05, X[i].shape)
            gi, ni = idx_list[i]
            additions.setdefault(gi, []).append((c, x_syn.astype(np.float32), ni))
        stats[c] = f"+{n_syn}"

    if not additions:
        log.info("SMOTE: các lớp đã cân bằng -> giữ nguyên")
        return train

    # Thay graph chủ bằng bản augmented
    out = list(train)
    n_new_nodes = 0
    for gi, items in additions.items():
        g = train[gi]
        x, ei, ea, y = g.x, g.edge_index, g.edge_attr, g.y
        n0 = x.shape[0]
        new_x, new_y, add_src, add_dst, add_ea = [], [], [], [], []
        for off, (c, x_syn, ni) in enumerate(items):
            new_id = n0 + off
            new_x.append(torch.from_numpy(x_syn))
            new_y.append(c)
            # copy cạnh của node nguồn ni
            m_out = (ei[0] == ni).nonzero(as_tuple=True)[0]
            m_in = (ei[1] == ni).nonzero(as_tuple=True)[0]
            for e in m_out.tolist():
                add_src.append(new_id); add_dst.append(int(ei[1, e]))
                add_ea.append(ea[e])
            for e in m_in.tolist():
                add_src.append(int(ei[0, e])); add_dst.append(new_id)
                add_ea.append(ea[e])
            if len(m_out) + len(m_in) == 0:      # node cô lập -> self-loop
                add_src.append(new_id); add_dst.append(new_id)
                add_ea.append(torch.zeros(ea.shape[1]))
        x2 = torch.cat([x, torch.stack(new_x)], dim=0)
        y2 = torch.cat([y, torch.tensor(new_y, dtype=torch.long)])
        ei2 = torch.cat([ei, torch.tensor([add_src, add_dst], dtype=torch.long)], dim=1)
        ea2 = torch.cat([ea, torch.stack(add_ea)], dim=0)
        from torch_geometric.data import Data
        g2 = Data(x=x2, edge_index=ei2, edge_attr=ea2, y=y2)
        # SỬA LỖI: bản cũ dựng Data mới và LÀM MẤT mọi thuộc tính phụ
        # (sim_id, time_win, src_file...). Khi một batch trộn graph đã
        # augment với graph gốc, PyG collate ném KeyError vì thuộc tính
        # chỉ có ở một phía. Lỗi này ảnh hưởng CẢ nhánh iotrpl (src_file).
        for k, v in g.items():
            if k not in ("x", "edge_index", "edge_attr", "y"):
                g2[k] = v
        out[gi] = g2
        n_new_nodes += len(items)

    log.info("SMOTE %s -> +%d node tổng hợp trong %d graph (tổng graph không đổi: %d)",
             stats, n_new_nodes, len(additions), len(out))
    return out


def make_loaders(train, val, test, batch_size=32, sampler=None):
    """Tạo DataLoader. Nếu có sampler thì dùng thay cho shuffle."""
    if sampler is not None:
        tr = DataLoader(train, batch_size=batch_size, sampler=sampler)
    else:
        tr = DataLoader(train, batch_size=batch_size, shuffle=True)
    return (
        tr,
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def build_weighted_sampler(train, num_classes, alpha=0.5, seed=42, cap=5.0):
    """LẤY MẪU LẠI CÓ TRỌNG SỐ Ở MỨC ĐỒ THỊ — KHÔNG sinh dữ liệu giả.

    ═══ VÌ SAO CẦN, DÙ ĐÃ CÓ CB-Focal ═══
    CB-Focal chỉ tăng TRỌNG SỐ khi mẫu CÓ MẶT trong batch — nó không tạo ra sự
    hiện diện. Đo trên phân bố thật của RADAR (18.266 đồ thị, batch 32):

        replay    ~286/18266 đồ thị chứa  -> 60,3% số batch KHÔNG có mẫu
        blackhole ~302/18266 đồ thị chứa  -> 58,7% số batch KHÔNG có mẫu
        delayed   ~619/18266 đồ thị chứa  -> 33,2% số batch KHÔNG có mẫu

    Hơn một nửa số bước cập nhật hoàn toàn không thấy lớp hiếm nhất. Trọng số
    loss dù lớn đến đâu cũng không sửa được điều đó.

    ═══ CÁCH LÀM ═══
    Lấy mẫu ĐỒ THỊ THẬT có hoàn lại, với xác suất tỉ lệ nghịch với độ hiếm của
    lớp hiếm nhất mà đồ thị đó chứa:

        w(G) = ( N / n_{c*(G)} ) ^ alpha ,   c*(G) = lớp hiếm nhất trong G

    Mọi mẫu đều là DỮ LIỆU THẬT — chỉ thay đổi TẦN SUẤT XUẤT HIỆN, không nội
    suy, không bịa đặc trưng. Khác hoàn toàn SMOTE.

    ═══ CHUẨN HOÁ CÓ TRẦN (cap) ═══
    Trọng số thô trải quá rộng: alpha=0,5 cho tỉ lệ max/min = 23 lần, alpha=1,0
    cho 529 lần — vài đồ thị Normal gần như KHÔNG BAO GIỜ được lấy mẫu, làm hỏng
    biểu diễn lớp đa số. Vì vậy CHẶN TRẦN tỉ lệ trước khi chuẩn hoá:

        w <- clip( w , w_min , cap * w_min )   rồi   w <- w / sum(w)

    cap = 5 (mặc định): lớp hiếm được ưu tiên tối đa 5 lần, Normal vẫn chiếm
    ~1,3% tỉ trọng lấy mẫu — đủ để giữ biểu diễn. cap = 0 để tắt trần.

    ═══ THAM SỐ alpha ═══
    alpha = 0   -> không đổi (đồng đều)
    alpha = 0.5 -> căn bậc hai, ÔN HOÀ (mặc định)
    alpha = 1.0 -> cân bằng hoàn toàn, dễ bù QUÁ ĐÀ khi đã có CB-Focal

    Chọn 0.5 vì CB-Focal đã bù một phần: hai cơ chế cộng dồn theo cấp số nhân,
    dùng alpha = 1 cùng lúc sẽ over-correct (đúng lỗi SMOTE đã mắc).
    """
    from torch.utils.data import WeightedRandomSampler

    counts = torch.zeros(num_classes)
    for g in train:
        for v, c in zip(*torch.unique(g.y, return_counts=True)):
            counts[int(v)] += int(c)
    counts = counts.clamp_min(1.0)
    N = float(counts.sum())

    w = []
    for g in train:
        present = [int(v) for v in torch.unique(g.y) if int(v) != 0]
        if not present:
            rare_n = counts[0]                      # đồ thị toàn Normal
        else:
            rare_n = min(counts[c] for c in present)
        w.append(float((N / rare_n) ** alpha))

    w_t = torch.tensor(w, dtype=torch.double)
    # CHUẨN HOÁ CÓ TRẦN: chặn tỉ lệ max/min để lớp hiếm không vọt quá xa
    if cap and cap > 0:
        lo = w_t.min()
        w_t = torch.clamp(w_t, min=lo, max=lo * float(cap))
    w_t = w_t / w_t.sum()
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(w_t, num_samples=len(train),
                                    replacement=True, generator=gen)

    # Nhật ký: tần suất KỲ VỌNG mỗi đồ thị được thấy trong một epoch
    p = (w_t / w_t.sum()).numpy()
    exp_seen = p * len(train)
    log.info("Sampler có trọng số (alpha=%.2f, cap=%s, KHÔNG sinh dữ liệu giả): "
             "tần suất kỳ vọng min=%.2f max=%.2f lần/epoch (tỉ lệ %.1f)",
             alpha, cap if cap else "tắt", exp_seen.min(), exp_seen.max(),
             exp_seen.max() / max(exp_seen.min(), 1e-9))
    return sampler


def compute_class_weights(train, num_classes):
    """Trọng số lớp nghịch tần suất — cho Weighted CrossEntropy."""
    counts = torch.zeros(num_classes)
    for g in train:
        for c in g.y:
            counts[int(c)] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
