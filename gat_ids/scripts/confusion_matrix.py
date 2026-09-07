"""MA TRẬN NHẦM LẪN — xác định chính xác lớp nào bị nhầm sang lớp nào.

Chạy:
    python scripts/confusion_matrix.py --ckpt-dir ck_L3_nomainres
    python scripts/confusion_matrix.py --ckpt-dir ck_L3 --model EdgeGAT --normalize row

VÌ SAO CẦN: hình t-SNE chỉ cho thấy các lớp CHỒNG LẤN, không cho biết chiều
nhầm lẫn. Ví dụ lớp `rank` có recall 0,412 — hình t-SNE cho thấy nó vỡ thành
nhiều mảnh, nhưng không nói 59% node còn lại bị gán sang lớp NÀO. Ma trận nhầm
lẫn trả lời chính xác câu hỏi đó, và đó mới là thứ định hướng cải tiến.

Sinh ra:
  - Hình ma trận nhầm lẫn chuẩn hoá theo hàng (mỗi hàng cộng = 100%)
  - Bảng TOP các cặp nhầm lẫn mạnh nhất, in ra màn hình
  - Tệp CSV ma trận thô để đưa vào phụ lục luận văn
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.common import get_logger, get_device, set_seed
from src.models.edge_gat import EdgeAwareGAT

log = get_logger()


def load_edgegat(ckpt_dir, device):
    d = os.path.join(ckpt_dir, "EdgeGAT")
    if not os.path.isdir(d):
        d = ckpt_dir
    with open(os.path.join(d, "model_config.json"), encoding="utf-8") as f:
        mc = json.load(f)
    m = EdgeAwareGAT(
        in_dim=mc["in_dim"], edge_dim=mc["edge_dim"], hidden_dim=mc["hidden_dim"],
        out_dim=mc["out_dim"], num_classes=mc["num_classes"], heads=mc["heads"],
        dropout=mc["dropout"], use_edge_features=mc["use_edge_features"],
        use_dyt=mc.get("use_dyt", True),
        use_input_skip=mc.get("use_input_skip", True),
        skip_mode=mc.get("skip_mode", "raw"),
        num_layers=mc.get("num_layers", 2),
        layer_residual=mc.get("layer_residual", False),
        use_main_residual=mc.get("use_main_residual", True),
        classifier_type=mc.get("classifier_type", "linear"),
        clf_hidden=mc.get("clf_hidden", 128))
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=device))
    m.to(device).eval()
    return m, mc


@torch.no_grad()
def predict_all(model, graphs, device):
    ys, ps = [], []
    for g in graphs:
        z = model(g.x.to(device), g.edge_index.to(device),
                  g.edge_attr.to(device) if g.edge_attr is not None else None)
        ps.append(z.argmax(1).cpu().numpy())
        ys.append(g.y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def plot_cm(cm_norm, cm_raw, names, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = len(names)
    fig, ax = plt.subplots(figsize=(max(9, C * 0.62), max(7.5, C * 0.55)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.2)
    ax.set_yticklabels(names, fontsize=8.2)
    ax.set_xlabel("Lớp dự đoán", fontsize=10)
    ax.set_ylabel("Lớp thật", fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    for i in range(C):
        for j in range(C):
            v = cm_norm[i, j]
            if v < 0.005:
                continue
            # in phần trăm; ô đường chéo và ô đậm dùng chữ trắng cho dễ đọc
            ax.text(j, i, "%.0f" % (100 * v), ha="center", va="center",
                    fontsize=7.2 if C > 12 else 8.4,
                    color="white" if v > 0.55 else "#333333",
                    fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Tỉ lệ theo hàng (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="confusion_matrix.png")
    ap.add_argument("--csv", default="confusion_matrix.csv")
    ap.add_argument("--top", type=int, default=12,
                    help="Số cặp nhầm lẫn mạnh nhất cần in")
    args = ap.parse_args()

    set_seed(42)
    device = get_device("auto")
    model, mc = load_edgegat(args.ckpt_dir, device)
    log.info("Đã nạp EdgeGAT | %d tầng | classifier=%s | %d tham số",
             mc.get("num_layers", 2), mc.get("classifier_type", "linear"),
             model.count_parameters())

    graphs = torch.load(os.path.join(args.data_dir, "%s.pt" % args.split),
                        weights_only=False)
    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["class_names"]
    C = len(names)

    y, p = predict_all(model, graphs, device)
    cm = np.zeros((C, C), dtype=np.int64)
    for a, b in zip(y, p):
        cm[int(a), int(b)] += 1

    row = cm.sum(1, keepdims=True).clip(min=1)
    cm_n = cm / row

    # ── Bảng cặp nhầm lẫn mạnh nhất ──
    pairs = []
    for i in range(C):
        for j in range(C):
            if i != j and cm[i, j] > 0:
                pairs.append((cm_n[i, j], cm[i, j], names[i], names[j], int(cm[i].sum())))
    pairs.sort(reverse=True)

    log.info("\n── TOP %d CẶP NHẦM LẪN (theo tỉ lệ của lớp thật) ──", args.top)
    log.info("   %-22s → %-22s %8s %10s", "lớp THẬT", "bị đoán thành", "số node", "% của lớp")
    for r, n, a, b, tot in pairs[:args.top]:
        log.info("   %-22s → %-22s %8d %9.1f%%", a, b, n, 100 * r)

    # ── Lớp nào bị phân tán nhiều nhất ──
    log.info("\n── ĐỘ PHÂN TÁN CỦA TỪNG LỚP THẬT ──")
    log.info("   %-22s %9s %10s %s", "lớp", "recall", "số lớp bị", "lớp hút nhiều nhất")
    for i in np.argsort(np.diag(cm_n)):
        tot = cm[i].sum()
        if tot == 0:
            continue
        wrong = [(cm_n[i, j], names[j]) for j in range(C) if j != i and cm[i, j] > 0]
        wrong.sort(reverse=True)
        top = ", ".join("%s %.0f%%" % (nm, 100 * v) for v, nm in wrong[:2]) or "—"
        log.info("   %-22s %9.3f %10d   %s", names[i], cm_n[i, i], len(wrong), top)

    np.savetxt(args.csv, cm, fmt="%d", delimiter=",",
               header=",".join(names), comments="")
    plot_cm(cm_n, cm, names, args.out,
            "Ma trận nhầm lẫn — EdgeGAT %d tầng, classifier %s (chuẩn hoá theo hàng, %%)"
            % (mc.get("num_layers", 2), mc.get("classifier_type", "linear")))
    log.info("\nĐã lưu hình -> %s", args.out)
    log.info("Đã lưu ma trận thô -> %s", args.csv)
    log.info("\nCÁCH ĐỌC: mỗi HÀNG là một lớp thật, cộng lại bằng 100%%. "
             "Ô đường chéo là recall. Ô ngoài đường chéo cho biết node của lớp "
             "đó bị gán nhầm sang lớp nào.")


if __name__ == "__main__":
    main()
