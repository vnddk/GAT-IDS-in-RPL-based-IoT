#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ata.py — Huấn luyện ATAEdgeGAT (EdgeGAT + Adaptive Temporal Alignment)
============================================================================

Hiện thực Thuật toán 1 của Phan và cộng sự (2026) cho bài toán phân loại nút
trên đồ thị DODAG:

    1.  Sắp tập huấn luyện theo thứ tự thời gian
    2.  Chia thành K miền thời gian liền kề
    3.  Mỗi bước: lấy MỘT lô từ MỖI miền (lấy mẫu cân bằng theo miền)
    4.  Tính L_cls trên từng miền và L_trans trên từng cặp miền
    5.  L_ATA = Σ_k L_cls^(k) + (λ_d/|P|) Σ_(a,b) L_trans^(a,b)
    6.  Chọn điểm lưu theo macro-F1 trên tập kiểm định

KIẾN TRÚC KHÔNG ĐỔI so với EdgeGAT. ATA chỉ thay đổi HÀM MỤC TIÊU và CÁCH
LẤY MẪU, nên mọi chênh lệch đo được quy về đúng phần đóng góp của ATA chứ
không lẫn với khác biệt kiến trúc. Checkpoint lưu ra tương thích hoàn toàn
với test_all.py.

Cách dùng
---------
    # 1) Chuẩn bị dữ liệu THEO THỨ TỰ THỜI GIAN (bắt buộc để ATA có nghĩa)
    python scripts/convert_data.py --dataset radar --data-dir data/radar \\
           --split chrono --alpha 1.0 --out-dir data/radar_chrono

    # 2) Huấn luyện ATAEdgeGAT
    python scripts/train_ata.py --data-dir data/radar_chrono \\
           --K 3 --lambda-d 0.1 --disc coral --epochs 120 \\
           --ckpt-dir ck_ata

    # 3) Đánh giá (dùng chung test_all.py)
    python scripts/test_all.py --data-dir data/radar_chrono --ckpt-dir ck_ata

Quét siêu tham số theo bài báo:
    --K {2,3,4,5}          số miền thời gian
    --lambda-d {0.01,0.1,0.5,1.0}   trọng số căn chỉnh
    --disc {coral,mmd,cosine}
    --domain-strategy {quantile,tdc}
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils.common import load_config, set_seed, get_device, get_logger   # noqa: E402
from src.models.edge_gat import build_model                                  # noqa: E402
from src.utils.engine import make_criterion, evaluate, EMA                   # noqa: E402
from src.utils.common import EarlyStopping                                   # noqa: E402
from src.utils.metrics import format_report                                  # noqa: E402
from src.utils.ata import (assign_temporal_domains, DISCREPANCY,             # noqa: E402
                           temporal_leakage_gap)

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints/ATAEdgeGAT")
    ap.add_argument("--arch", default="edgegat",
                    choices=["edgegat", "negat", "mlp", "gcn", "egraphsage"],
                    help="Kiến trúc nền để áp ATA. edgegat = mô hình một kênh "
                         "hiện tại; negat = kiến trúc hai kênh Node/Edge. ATA "
                         "chỉ thay HÀM MỤC TIÊU và CÁCH LẤY MẪU, không đụng tới "
                         "kiến trúc, nên cùng một mã huấn luyện dùng được cho cả hai.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    # ── siêu tham số ATA ──
    ap.add_argument("--K", type=int, default=3,
                    help="số miền thời gian. K=1 thoái hoá về ERM thông thường "
                         "và không tạo áp lực xuyên giai đoạn nào.")
    ap.add_argument("--lambda-d", type=float, default=0.1,
                    help="trọng số căn chỉnh. λ→0 trở lại bộ phân loại đa đoạn "
                         "thông thường; λ quá lớn điều chuẩn quá mức và làm xói "
                         "mòn khả năng tách lớp.")
    ap.add_argument("--disc", default="coral", choices=["coral", "mmd", "cosine"])
    ap.add_argument("--domain-strategy", default="quantile", choices=["quantile", "tdc"])
    # ── siêu tham số riêng của NE-GAT ──
    ap.add_argument("--edge-score", default="full", choices=["full", "edge_only"],
                    help="NE-GAT: cách tính điểm chú ý của kênh cạnh")
    ap.add_argument("--edge-agg", default="mean_deg", choices=["mean_deg", "mean"],
                    help="NE-GAT: mean = trung bình có trọng số (MẤT thông tin "
                         "số lượng liên kết); mean_deg = bổ sung thành phần theo bậc")
    ap.add_argument("--no-edge-update", action="store_true",
                    help="NE-GAT: tắt cập nhật trạng thái cạnh qua từng tầng")
    ap.add_argument("--edge-latent", type=int, default=24)
    ap.add_argument("--node-heads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="kích thước lô MỖI MIỀN (mặc định lấy từ config)")
    ap.add_argument("--cls-reduction", default="mean", choices=["mean", "sum"],
                    help="Cách gộp L_cls trên K miền. Bài báo viết TỔNG (sum), "
                         "nhưng khi so ATA (K=3) với đối chứng (K=1) thì tổng "
                         "làm gradient lớn gấp K lần, tức đổi ngầm tốc độ học "
                         "và tạo yếu tố nhiễu cho phép so sánh. mean (mặc định) "
                         "chia cho K để loại yếu tố đó; sum tái lập đúng Eq.(8).")
    ap.add_argument("--align-normal-only", action="store_true",
                    help="chỉ căn chỉnh trên nút lớp Normal. Lý do có tuỳ chọn "
                         "này: phân bố lớp tấn công thay đổi theo thiết kế mô "
                         "phỏng chứ không phải do dịch chuyển thời gian, nên ép "
                         "chúng giống nhau giữa các giai đoạn có thể phản tác dụng.")
    a = ap.parse_args()

    set_seed(a.seed)
    cfg = load_config(a.config)
    cfg["train"]["epochs"] = a.epochs
    cfg["train"]["ckpt_dir"] = a.ckpt_dir
    bs = a.batch_size or cfg["train"]["batch_size"]
    device = get_device(cfg["device"])
    os.makedirs(a.ckpt_dir, exist_ok=True)

    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    ld = lambda n: torch.load(os.path.join(a.data_dir, n), weights_only=False)
    train_g, val_g, test_g = ld("train.pt"), ld("val.pt"), ld("test.pt")
    log.info("Tải: train=%d val=%d test=%d | d_n=%d d_e=%d | %d lớp",
             len(train_g), len(val_g), len(test_g),
             meta["n_node_features"], meta["n_edge_features"], meta["num_classes"])

    # ── chẩn đoán rò rỉ thời gian ──
    gap = temporal_leakage_gap(train_g, test_g)
    log.info("Khoảng cách thời gian gần nhất trung bình δ̄ = %.3f cửa sổ", gap)
    if gap < 2.0:
        log.warning("δ̄ RẤT NHỎ: mẫu kiểm thử nằm gần như liền kề mẫu huấn "
                    "luyện về thời gian. Đây là dấu hiệu RÒ RỈ THỜI GIAN — phép "
                    "chia hiện tại nhiều khả năng là ngẫu nhiên chứ không theo "
                    "thứ tự thời gian. ATA được thiết kế cho chế độ tiến-theo-"
                    "thời-gian, nên chạy trên dữ liệu chia ngẫu nhiên sẽ KHÔNG "
                    "đo được đúng thứ nó nhắm tới. Dùng convert_data.py "
                    "--split chrono.")

    # ── dựng miền thời gian ──
    dom = assign_temporal_domains(train_g, K=a.K, strategy=a.domain_strategy,
                                  disc=a.disc)
    loaders = []
    for k in range(a.K):
        sub = [g for g, d in zip(train_g, dom) if d == k]
        if not sub:
            log.error("Miền %d rỗng — giảm --K.", k); return 1
        loaders.append(DataLoader(sub, batch_size=bs, shuffle=True, drop_last=False))
    val_loader = DataLoader(val_g, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_g, batch_size=bs, shuffle=False)

    # ── mô hình và hàm mất mát ──
    if a.arch in ("mlp", "gcn", "egraphsage"):
        # ATA là kỹ thuật dựa trên HÀM MẤT MÁT nên chỉ áp dụng được cho mô hình
        # huấn luyện bằng hạ gradient. Các baseline dạng CÂY (rừng ngẫu nhiên,
        # XGBoost, LightGBM) KHÔNG dùng được ATA — chúng vẫn được huấn luyện
        # bằng train_all.py và so sánh trên cùng tập chia theo thời gian.
        from src.models.gnn_baselines import build_baseline
        _map = {"mlp": "mlp", "gcn": "gcn", "egraphsage": "edge_graphsage"}
        model = build_baseline(_map[a.arch], meta, cfg).to(device)
        _tag = f"ATA-{a.arch.upper()}"
    elif a.arch == "negat":
        cfg["model"]["edge_score_mode"] = a.edge_score
        cfg["model"]["edge_agg"] = a.edge_agg
        cfg["model"]["edge_update"] = not a.no_edge_update
        cfg["model"]["edge_latent"] = a.edge_latent
        cfg["model"]["node_heads"] = a.node_heads
        from src.models.ne_gat import build_negat
        model = build_negat(cfg, meta).to(device)
        _tag = "ATA-NEGAT"
    else:
        model = build_model(cfg, meta).to(device)
        _tag = "ATAEdgeGAT"
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s: %d tham số | K=%d λ_d=%.3g disc=%s chiến lược=%s",
             _tag, n_par, a.K, a.lambda_d, a.disc, a.domain_strategy)
    crit = make_criterion(cfg, train_g, meta["num_classes"], device)
    disc_fn = DISCREPANCY[a.disc]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    T = a.epochs
    Tw = cfg["train"].get("warmup_epochs", 6)
    mr = cfg["train"].get("min_lr_ratio", 0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / max(Tw, 1) if ep < Tw
        else mr + (1 - mr) * 0.5 * (1 + np.cos(np.pi * (ep - Tw) / max(T - Tw, 1))))
    gclip = cfg["train"].get("grad_clip", 1.0)

    # ── Ba cơ chế dưới đây SAO CHÉP nguyên từ train_centralized ──
    # Nếu thiếu chúng, ATAEdgeGAT và EdgeGAT sẽ khác nhau ở BA điểm ngoài ATA
    # (EMA, làm mượt val, dừng sớm), khiến chênh lệch đo được không quy được
    # về riêng ATA. Đây là điều kiện bắt buộc để phép so sánh có kiểm soát.
    t = cfg["train"]
    use_ema = t.get("use_ema", True)
    ema = EMA(model, decay=t.get("ema_decay", 0.995)) if use_ema else None
    eval_model = copy.deepcopy(model) if ema is not None else model
    stopper = EarlyStopping(patience=t.get("early_stopping_patience", 30), mode="max")
    smooth_beta = t.get("val_smooth_beta", 0.6)
    ema_val = None

    n_pairs = a.K * (a.K - 1) // 2
    best, best_ep, hist = -1.0, -1, []
    ck = os.path.join(a.ckpt_dir, "best_model.pt")

    for ep in range(1, a.epochs + 1):
        model.train()
        iters = [iter(dl) for dl in loaders]
        n_steps = max(len(dl) for dl in loaders)
        s_cls = s_ali = 0.0
        for _ in range(n_steps):
            opt.zero_grad()
            feats, loss_cls = [], 0.0
            for k in range(a.K):
                try:
                    batch = next(iters[k])
                except StopIteration:            # miền ngắn hơn -> lặp lại
                    iters[k] = iter(loaders[k]); batch = next(iters[k])
                batch = batch.to(device)
                logits, z = model(batch.x, batch.edge_index, batch.edge_attr,
                                  return_feat=True)
                loss_cls = loss_cls + crit(logits, batch.y)
                if a.align_normal_only:
                    m = (batch.y == 0)
                    feats.append(z[m] if int(m.sum()) >= 2 else z)
                else:
                    feats.append(z)

            loss_ali = z.new_zeros(())
            if n_pairs > 0 and a.lambda_d > 0:
                for i in range(a.K):
                    for j in range(i + 1, a.K):
                        loss_ali = loss_ali + disc_fn(feats[i], feats[j])
                loss_ali = loss_ali / n_pairs

            if a.cls_reduction == "mean":
                loss_cls = loss_cls / a.K          # loại yếu tố nhiễu do K
            loss = loss_cls + a.lambda_d * loss_ali
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
            opt.step()
            if ema is not None:
                ema.update(model)
            s_cls += float(loss_cls); s_ali += float(loss_ali)
        sched.step()

        if ema is not None:
            ema.copy_to(eval_model)
        vm = evaluate(eval_model if ema is not None else model, val_loader,
                      device, meta["num_classes"], meta["class_names"])
        f1 = vm["macro_f1"]; hist.append(f1)
        ema_val = f1 if ema_val is None else smooth_beta * ema_val + (1 - smooth_beta) * f1
        is_best = stopper.step(ema_val)
        if is_best:
            best, best_ep = ema_val, ep
            torch.save((eval_model if ema is not None else model).state_dict(), ck)
        if ep % 5 == 0 or ep == 1 or is_best:
            log.info("Epoch %3d | L_cls %.4f | L_align %.4f | val %.4f (mượt %.4f)%s",
                     ep, s_cls / n_steps, s_ali / n_steps, f1, ema_val,
                     "  <-- best" if is_best else "")
        if stopper.should_stop:
            log.info("Dừng sớm tại epoch %d (không cải thiện %d epoch).",
                     ep, stopper.patience)
            break

    model.load_state_dict(torch.load(ck, map_location=device))
    tm = evaluate(model, test_loader, device, meta["num_classes"],
                  meta["class_names"], with_binary=True)
    log.info("== KẾT QUẢ TEST (%s) ==\n%%s" % _tag,
             format_report(tm, meta["class_names"]))
    from src.utils.binary_eval import format_binary
    log.info("\n%s", format_binary(tm["binary"]))
    log.info("Best val macro-F1 (đã làm mượt) = %.4f tại epoch %d", best, best_ep)

    mc = {"in_dim": meta["n_node_features"], "edge_dim": meta["n_edge_features"],
          "num_classes": meta["num_classes"],
          "hidden_dim": cfg["model"]["hidden_dim"], "out_dim": cfg["model"]["out_dim"],
          "heads": cfg["model"]["heads"], "dropout": cfg["model"]["dropout"],
          "use_edge_features": cfg["model"]["use_edge_features"],
          "use_dyt": cfg["model"].get("use_dyt", True),
          "use_input_skip": cfg["model"].get("use_input_skip", True),
          "skip_mode": cfg["model"].get("skip_mode", "raw"),
          "num_layers": cfg["model"].get("num_layers", 2),
          "layer_residual": cfg["model"].get("layer_residual", False),
          "use_main_residual": cfg["model"].get("use_main_residual", True),
          "classifier_type": cfg["model"].get("classifier_type", "mlp"),
          "clf_hidden": cfg["model"].get("clf_hidden", 128),
          "class_names": meta["class_names"],
          "model_type": {"negat": "NE-GAT", "mlp": "MLP", "gcn": "GCN",
                         "egraphsage": "E-GraphSAGE"}.get(a.arch, "EdgeGAT"),
          "edge_latent": a.edge_latent, "node_heads": a.node_heads,
          "edge_score_mode": a.edge_score, "edge_agg": a.edge_agg,
          "edge_update": not a.no_edge_update,
          "ata": {"K": a.K, "lambda_d": a.lambda_d, "disc": a.disc,
                  "strategy": a.domain_strategy,
                  "align_normal_only": bool(a.align_normal_only),
                  "cls_reduction": a.cls_reduction,
                  "delta_bar": gap}}
    json.dump(mc, open(os.path.join(a.ckpt_dir, "model_config.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"val_macro_f1": hist, "best": best, "best_epoch": best_ep,
               "test": {k: v for k, v in tm.items() if not isinstance(v, dict)}},
              open(os.path.join(a.ckpt_dir, "ata_history.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Đã lưu -> %s", a.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
