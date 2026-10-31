"""HIỆU CHỈNH LOGIT SAU HUẤN LUYỆN — nâng recall lớp hiếm, không cần train lại.

Chạy:
    python scripts/calibrate_logits.py --ckpt-dir ck_hybrid

VÌ SAO CẦN (chẩn đoán từ log 60 epoch):
    9/16 lớp có Precision > Recall, trung bình chênh +0,135. Ví dụ:
        delayed_reply  P=0,458  R=0,101   (chênh +0,357)
        sybil          P=0,872  R=0,562   (chênh +0,310)
        hello_flood    P=1,000  R=0,701   (chênh +0,299)
    Nghĩa là mô hình QUÁ THẬN TRỌNG: khi thấy tấn công thì đoán gần như đúng,
    nhưng bỏ sót rất nhiều. Với F1, khi P >> R thì đánh đổi bớt precision lấy
    recall sẽ LÀM TĂNG F1. Ước lượng bảo thủ: +0,009 macro-F1, riêng
    delayed_reply +0,05.

CÁCH LÀM (Menon và cs., ICLR 2021 — Logit Adjustment):
    Lúc suy luận, trừ đi một lượng tỉ lệ với log tần suất lớp:
        ŷ = argmax_c ( z_c − τ · log π_c )
    π_c là tần suất lớp c trong tập huấn luyện. Lớp càng hiếm thì log π_c càng
    âm, nên bị trừ một số âm = ĐƯỢC CỘNG thêm -> dễ được chọn hơn.
    τ = 0 là mô hình gốc; τ = 1 là hiệu chỉnh đầy đủ theo lý thuyết Bayes.

QUAN TRỌNG VỀ PHƯƠNG PHÁP:
    τ được dò trên tập VALIDATION, sau đó áp cố định cho tập TEST. Tuyệt đối
    không dò τ trên test — làm vậy là rò rỉ và con số sẽ không bảo vệ được.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.common import get_logger, get_device, set_seed
from src.utils.metrics import compute_metrics, format_report
from src.models.edge_gat import EdgeAwareGAT

log = get_logger()


def load_model(ckpt_dir, device):
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
        skip_mode=mc.get("skip_mode", "raw"))
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=device))
    m.to(device).eval()
    return m, mc


@torch.no_grad()
def collect_logits(model, graphs, device):
    Z, Y = [], []
    for g in graphs:
        z = model(g.x.to(device), g.edge_index.to(device),
                  g.edge_attr.to(device) if g.edge_attr is not None else None)
        Z.append(z.cpu().numpy()); Y.append(g.y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def macro_f1(y, pred, C):
    f = []
    for c in range(C):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fn == 0:
            continue                      # lớp không có mẫu -> bỏ khỏi mẫu số
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f)) if f else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tau-grid",
                    default="-0.6,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,0.8,1.0",
                    help="Dò cả tau ÂM: khi mô hình BÙ QUÁ ĐÀ (precision lớp hiếm "
                         "thấp, recall cao) thì tau<0 kéo về, tăng accuracy.")
    ap.add_argument("--metric", default="macro_f1",
                    choices=["macro_f1", "accuracy", "balanced"],
                    help="Tiêu chí chọn tau: macro_f1 | accuracy | balanced (trung bình 2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device("auto")
    model, mc = load_model(args.ckpt_dir, device)
    C = mc["num_classes"]

    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cn = meta["class_names"]

    tr = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=False)
    va = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=False)
    te = torch.load(os.path.join(args.data_dir, "test.pt"), weights_only=False)

    # tần suất lớp trên tập HUẤN LUYỆN (gốc, chưa resample)
    cnt = np.zeros(C)
    for g in tr:
        for v, c in zip(*np.unique(g.y.numpy(), return_counts=True)):
            cnt[int(v)] += c
    prior = np.log((cnt + 1) / (cnt.sum() + C))

    log.info("Thu logits trên val và test...")
    Zv, Yv = collect_logits(model, va, device)
    Zt, Yt = collect_logits(model, te, device)

    taus = [float(x) for x in args.tau_grid.split(",")]
    log.info("── DÒ τ TRÊN TẬP VALIDATION ──")
    best_tau, best_f1 = 0.0, -1.0
    for t in taus:
        pv = (Zv - t * prior).argmax(1)
        mf = macro_f1(Yv, pv, C)
        acc = float((pv == Yv).mean())
        f = {"macro_f1": mf, "accuracy": acc,
             "balanced": 0.5 * (mf + acc)}[args.metric]
        mark = ""
        if f > best_f1:
            best_f1, best_tau, mark = f, t, "  <-- tốt nhất"
        log.info("   τ=%-6.2f  val macroF1=%.4f  acc=%.4f  tiêu chí(%s)=%.4f%s",
                 t, mf, acc, args.metric, f, mark)

    log.info("\n── ÁP τ=%.2f CHO TẬP TEST (τ chọn trên val, KHÔNG trên test) ──", best_tau)
    m0 = compute_metrics(Yt, Zt.argmax(1), C, cn)
    m1 = compute_metrics(Yt, (Zt - best_tau * prior).argmax(1), C, cn)
    log.info("   Trước hiệu chỉnh : acc=%.4f  macro-F1=%.4f", m0["accuracy"], m0["macro_f1"])
    log.info("   Sau  hiệu chỉnh : acc=%.4f  macro-F1=%.4f   (%+.4f)",
             m1["accuracy"], m1["macro_f1"], m1["macro_f1"] - m0["macro_f1"])

    log.info("\n── THAY ĐỔI F1 THEO TỪNG LỚP ──")
    log.info("   %-20s %8s %8s %8s", "lớp", "trước", "sau", "Δ")
    for c in cn:
        a = m0["per_class_f1"].get(c, 0.0)
        b = m1["per_class_f1"].get(c, 0.0)
        if a == 0 and b == 0:
            continue
        log.info("   %-20s %8.3f %8.3f %+8.3f", c, a, b, b - a)

    log.info("\n%s", format_report(m1, cn))
    with open("calibration_result.json", "w", encoding="utf-8") as f:
        json.dump({"best_tau": best_tau,
                   "test_macro_f1_before": m0["macro_f1"],
                   "test_macro_f1_after": m1["macro_f1"]}, f, indent=2)
    log.info("Đã lưu -> calibration_result.json")


if __name__ == "__main__":
    main()
