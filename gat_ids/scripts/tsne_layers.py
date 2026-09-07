"""TRỰC QUAN HOÁ t-SNE ĐẶC TRƯNG HỌC ĐƯỢC THEO TỪNG TẦNG

Sinh hình giống Fig. 13 của các bài báo IDS: chiếu biểu diễn ẩn xuống 2 chiều
bằng t-SNE, tô màu theo lớp thật, để thấy mô hình TÁCH LỚP tốt dần qua các tầng.

Chạy:
    python scripts/tsne_layers.py
    python scripts/tsne_layers.py --ckpt-dir checkpoints --n-samples 4000
    python scripts/tsne_layers.py --layers h1,h2,final --perplexity 30

Ý nghĩa ba tầng:
    h1     — sau tầng GAT 1 (144 chiều): đã trộn láng giềng 1-hop
    h2     — sau tầng GAT 2 (32 chiều) : đã trộn 2-hop, có residual từ h0
    final  — vector ngay TRƯỚC bộ phân loại: gồm h2 + nhánh JK + raw-skip
             (đây là biểu diễn mà lớp Linear cuối thực sự nhìn thấy)

Cách ĐỌC hình cho luận văn:
    • Cụm tách rời, ranh giới rõ  -> đặc trưng phân biệt tốt
    • Các lớp chồng lấn nhau      -> đó chính là các lớp bị nhầm trong ma trận
                                     nhầm lẫn; đối chiếu để giải thích F1 thấp
    • So h1 -> h2 -> final        -> nếu cụm tách rõ dần thì kiến trúc đang
                                     làm đúng việc; nếu h2 nhoè hơn h1 thì đó
                                     là DẤU HIỆU OVER-SMOOTHING

LƯU Ý PHƯƠNG PHÁP (nên ghi trong luận văn):
    t-SNE chỉ dùng để MÔ TẢ ĐỊNH TÍNH, không phải bằng chứng định lượng.
    Khoảng cách giữa các cụm trong hình t-SNE KHÔNG phản ánh khoảng cách thật;
    chỉ cấu trúc lân cận cục bộ là đáng tin. Mọi kết luận về hiệu năng phải
    dựa trên macro-F1 và ma trận nhầm lẫn.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.common import get_logger, get_device, set_seed
from src.models.edge_gat import EdgeAwareGAT

log = get_logger()

# Bảng màu tách biệt tốt cho tối đa 16 lớp
PALETTE = [
    "#7F7F7F", "#D62728", "#2CA02C", "#1F77B4", "#FF7F0E", "#9467BD",
    "#8C564B", "#E377C2", "#BCBD22", "#17BECF", "#393B79", "#B5CF6B",
    "#E7BA52", "#AD494A", "#A55194", "#6B6ECF",
]


def load_model(ckpt_dir, device):
    d = os.path.join(ckpt_dir, "EdgeGAT")
    if not os.path.isdir(d):
        d = ckpt_dir                      # cho phép trỏ thẳng vào thư mục model
    cfg_p = os.path.join(d, "model_config.json")
    w_p = os.path.join(d, "best_model.pt")
    if not (os.path.exists(cfg_p) and os.path.exists(w_p)):
        log.error("Thiếu %s hoặc %s — huấn luyện EdgeGAT trước.", cfg_p, w_p)
        return None, None
    with open(cfg_p, encoding="utf-8") as f:
        mc = json.load(f)
    # Tự nhận diện kiến trúc từ model_config. NE-GAT chỉ phơi ra biểu diễn
    # TRƯỚC bộ phân loại (khoá "final"), không có các tầng ẩn trung gian như
    # EdgeGAT, nên khi vẽ theo tầng chỉ có một bảng — điều này là đúng chứ
    # không phải lỗi, và cần nêu rõ khi trình bày.
    _mt = str(mc.get("model_type", "EdgeGAT")).lower()
    if _mt in ("ne-gat", "negat", "ne_gat"):
        import inspect
        from src.models.ne_gat import NEGAT
        ok = set(inspect.signature(NEGAT.__init__).parameters) - {"self"}
        kw = {k: v for k, v in mc.items() if k in ok}
        kw["verbose"] = False
        model = NEGAT(**kw)
        model.load_state_dict(torch.load(w_p, map_location=device))
        model.to(device).eval()
        return model, mc

    model = EdgeAwareGAT(
        in_dim=mc["in_dim"], edge_dim=mc["edge_dim"],
        hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
        num_classes=mc["num_classes"], heads=mc["heads"],
        dropout=mc["dropout"], use_edge_features=mc["use_edge_features"],
        use_dyt=mc.get("use_dyt", True),
        use_input_skip=mc.get("use_input_skip", True),
        skip_mode=mc.get("skip_mode", "raw"),
        num_layers=mc.get("num_layers", 2),
        layer_residual=mc.get("layer_residual", False),
        use_main_residual=mc.get("use_main_residual", True),
        classifier_type=mc.get("classifier_type", "linear"),
        clf_hidden=mc.get("clf_hidden", 64))
    model.load_state_dict(torch.load(w_p, map_location=device))
    model.to(device).eval()
    return model, mc


@torch.no_grad()
def collect_embeddings(model, graphs, device, layers):
    """Chạy suy luận, gom biểu diễn từng tầng + nhãn thật."""
    acc = {k: [] for k in layers}
    ys = []
    for g in graphs:
        emb = model(g.x.to(device), g.edge_index.to(device),
                    g.edge_attr.to(device) if g.edge_attr is not None else None,
                    return_embeddings=True)
        for k in layers:
            acc[k].append(emb[k].cpu().numpy())
        ys.append(g.y.numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}, np.concatenate(ys)


def stratified_subsample(y, n_total, seed=42, min_per_class=30):
    """Lấy mẫu phân tầng: bảo đảm lớp hiếm vẫn hiện diện đủ để nhìn thấy.

    t-SNE trên 70.000+ node vừa rất chậm vừa bị lớp Normal (87%) che lấp hoàn
    toàn. Lấy mẫu cân bằng hơn giúp NHÌN THẤY các lớp tấn công.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    quota = max(min_per_class, n_total // max(1, len(classes)))
    idx = []
    for c in classes:
        pool = np.where(y == c)[0]
        take = min(len(pool), quota)
        idx.append(rng.choice(pool, size=take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def run_tsne(X, perplexity, seed=42):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    # PCA sơ bộ khi số chiều lớn — khuyến nghị chuẩn của tác giả t-SNE,
    # vừa khử nhiễu vừa tăng tốc đáng kể.
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=seed).fit_transform(Xs)
    per = float(min(perplexity, max(5, (len(Xs) - 1) / 3)))
    ts = TSNE(n_components=2, perplexity=per, init="pca",
              learning_rate="auto", max_iter=1000, random_state=seed)
    return ts.fit_transform(Xs)


def plot_panels(embs2d, y, class_names, out_png, titles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(embs2d)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.8))
    if n == 1:
        axes = [axes]
    classes = np.unique(y)
    for ax, (name, Z) in zip(axes, embs2d.items()):
        for c in classes:
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.75,
                       c=PALETTE[int(c) % len(PALETTE)],
                       label=class_names[int(c)] if class_names else str(c),
                       edgecolors="none")
        ax.set_title(titles.get(name, name), fontsize=12)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15, linestyle=":")
    # chú giải chung, đặt ngoài để không che dữ liệu
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=8, markerscale=2.0,
               frameon=True, borderaxespad=0.5)
    fig.suptitle("Đặc trưng ẩn của EdgeGAT trực quan hoá bằng t-SNE "
                 "(tô màu theo lớp thật)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--layers", default="auto",
                    help="Danh sách tầng, vd h1,h2,h3,final. 'auto' = mọi tầng + final")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tsne_layers.png")
    ap.add_argument("--exclude-normal", action="store_true",
                    help="Bỏ lớp Normal để nhìn rõ cấu trúc giữa các lớp tấn công")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device("auto")
    layers = None   # xác định sau khi biết num_layers từ model_config

    model, mc = load_model(args.ckpt_dir, device)
    if model is None:
        return
    log.info("Đã nạp EdgeGAT | %d tầng GAT | skip_mode=%s | %d tham số",
             mc.get("num_layers", 2), mc.get("skip_mode", "raw"),
             model.count_parameters())

    graphs = torch.load(os.path.join(args.data_dir, f"{args.split}.pt"),
                        weights_only=False)
    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta.get("class_names")

    L_cfg = int(mc.get("num_layers", 2))
    if args.layers.strip().lower() in ("auto", ""):
        layers = ["h%d" % i for i in range(1, L_cfg + 1)] + ["final"]
    else:
        layers = [t.strip() for t in args.layers.split(",") if t.strip()]
    log.info("Các tầng sẽ vẽ: %s", ", ".join(layers))
    log.info("Trích biểu diễn từ %d đồ thị (%s)...", len(graphs), args.split)
    embs, y = collect_embeddings(model, graphs, device, layers)
    log.info("Tổng %d node | số chiều: %s", len(y),
             {k: v.shape[1] for k, v in embs.items()})

    if args.exclude_normal:
        keep = y != 0
        y = y[keep]
        embs = {k: v[keep] for k, v in embs.items()}
        log.info("Đã bỏ lớp Normal -> còn %d node", len(y))

    idx = stratified_subsample(y, args.n_samples, args.seed)
    y_s = y[idx]
    log.info("Lấy mẫu phân tầng: %d node | phân bố: %s", len(idx),
             {int(c): int((y_s == c).sum()) for c in np.unique(y_s)})

    # Tiêu đề panel sinh động theo số tầng thực tế
    L = int(mc.get("num_layers", 2))
    titles = {}
    for i in range(1, L + 1):
        k = "h%d" % i
        if k in embs:
            nm = ("(%s) Tầng ẩn %d — sau GAT %d" % (chr(96 + i), i, i))
            titles[k] = "%s (%dD)" % (nm, embs[k].shape[1])
    if "final" in embs:
        titles["final"] = ("(%s) Tầng cuối — trước bộ phân loại (%dD)"
                           % (chr(96 + L + 1), embs["final"].shape[1]))
    if "logits" in embs:
        titles["logits"] = "Logits (%dD)" % embs["logits"].shape[1]

    embs2d = {}
    for k in layers:
        log.info("Chạy t-SNE cho '%s' (%d x %d)...", k, len(idx), embs[k].shape[1])
        embs2d[k] = run_tsne(embs[k][idx], args.perplexity, args.seed)

    plot_panels(embs2d, y_s, class_names, args.out, titles)
    log.info("Đã lưu hình -> %s", args.out)

    # Chỉ số định lượng bổ trợ: t-SNE là định tính, nên kèm số đo tách cụm
    try:
        from sklearn.metrics import silhouette_score
        log.info("── Điểm silhouette trên KHÔNG GIAN GỐC (cao = tách tốt) ──")
        for k in layers:
            sc = silhouette_score(embs[k][idx], y_s, metric="euclidean")
            log.info("   %-7s : %+.4f", k, sc)
        log.info("   (dùng số này trong luận văn, KHÔNG dùng khoảng cách đọc từ hình t-SNE)")
    except Exception as e:
        log.warning("Bỏ qua silhouette: %s", e)


if __name__ == "__main__":
    main()
