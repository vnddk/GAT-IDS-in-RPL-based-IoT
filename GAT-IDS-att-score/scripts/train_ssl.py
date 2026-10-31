#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ssl.py — Huấn luyện GAT-Transformer IDS theo hai giai đoạn
=================================================================

Giai đoạn 1 — TIỀN HUẤN LUYỆN TỰ GIÁM SÁT
    Chỉ dùng nút BÌNH THƯỜNG. Mục tiêu là tái tạo biểu diễn đồ thị qua bộ tự
    mã hoá có che. KHÔNG dùng nhãn tấn công dưới bất kỳ hình thức nào.
    Gradient của lỗi bình phương trung bình lan ngược qua CẢ Transformer LẪN
    bộ mã hoá đồ thị, nên biểu diễn được tối ưu trực tiếp cho tác vụ tái tạo.

    Đầu ra dùng được ngay như một bộ PHÁT HIỆN BẤT THƯỜNG nhị phân: điểm số
    là lỗi tái tạo, nút có lỗi cao bất thường bị gắn cờ.

Giai đoạn 2 — TINH CHỈNH CÓ GIÁM SÁT (tuỳ chọn)
    Gắn đầu phân loại lên biểu diễn đã học, huấn luyện trên dữ liệu có nhãn
    với hàm mất mát CB-Focal. Đây là bước BẮT BUỘC nếu cần phân loại nhiều
    lớp, vì một lỗi tái tạo vô hướng không thể sinh ra C lớp.

VÌ SAO TÁCH HAI GIAI ĐOẠN. Bài báo gốc chỉ có giai đoạn 1 vì bài toán của họ
là phát hiện bất thường nhị phân. Đề tài cần 16 lớp, nên giai đoạn 2 là bắt
buộc. Giá trị của giai đoạn 1 nằm ở chỗ nó học được phân bố của lưu lượng
bình thường mà KHÔNG tốn nhãn, nên có thể dùng nhiều dữ liệu chưa gán nhãn.

Cách dùng
---------
    # Giai đoạn 1 — tự giám sát, chỉ nút bình thường
    python scripts/train_ssl.py --data-dir data/radar_chrono --stage ssl \\
           --backbone negat --epochs 60 --ckpt-dir ck_ssl

    # Giai đoạn 2 — tinh chỉnh có giám sát từ checkpoint giai đoạn 1
    python scripts/train_ssl.py --data-dir data/radar_chrono --stage finetune \\
           --backbone negat --init-from ck_ssl --epochs 120 --ckpt-dir ck_gtids

    # Huấn luyện thẳng có giám sát, KHÔNG tiền huấn luyện (nhánh đối chứng)
    python scripts/train_ssl.py --data-dir data/radar_chrono --stage scratch \\
           --backbone negat --epochs 120 --ckpt-dir ck_gtids_scratch
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

from src.utils.common import load_config, set_seed, get_device, get_logger, EarlyStopping  # noqa: E402
from src.utils.engine import make_criterion, evaluate, EMA                                 # noqa: E402
from src.utils.metrics import format_report                                                # noqa: E402
from src.utils.binary_eval import binary_metrics, format_binary                            # noqa: E402
from src.models.gat_transformer import GATTransformerIDS                                   # noqa: E402

log = get_logger()


def filter_benign(graphs):
    """Giữ lại ĐỒ THỊ chỉ gồm nút bình thường.

    Lọc ở mức ĐỒ THỊ chứ không ở mức nút, vì bộ mã hoá đồ thị cần cấu trúc
    lân cận nguyên vẹn; xoá nút tấn công khỏi giữa một đồ thị sẽ phá vỡ tô-pô
    và tạo ra ngữ cảnh không tồn tại trong thực tế.
    """
    out = [g for g in graphs if int(g.y.max()) == 0]
    log.info("Lọc đồ thị THUẦN bình thường: %d / %d (%.1f%%)",
             len(out), len(graphs), 100.0 * len(out) / max(len(graphs), 1))
    if not out:
        log.error(
            "Không còn đồ thị nào sau khi lọc — không tiền huấn luyện được.\n"
            "  Nguyên nhân thường gặp: bộ dữ liệu mất cân bằng theo chiều "
            "NGƯỢC (gần như mọi cửa sổ đều chứa ít nhất một nút tấn công), "
            "đúng như bộ UOS và BoT-IoT.\n"
            "  Hai lối đi:\n"
            "    1. Giảm độ dài cửa sổ khi convert_data.py để sinh ra nhiều "
            "cửa sổ thuần bình thường hơn;\n"
            "    2. Bỏ giai đoạn SSL, chạy thẳng --stage scratch (có giám sát "
            "từ đầu) và so sánh với finetune trên bộ khác.")
    elif len(out) < 64:
        log.warning("Chỉ có %d đồ thị thuần bình thường — quá ít để tiền huấn "
                    "luyện có ý nghĩa. Kết quả SSL sẽ rất nhiễu; cân nhắc "
                    "--stage scratch.", len(out))
    return out


@torch.no_grad()
def collect_scores(model, loader, device, window, scope="batch"):
    model.eval()
    S, Y = [], []
    for b in loader:
        b = b.to(device)
        h, hr = model(b.x, b.edge_index, b.edge_attr, mode="ssl", window=window,
                      batch_vec=(b.batch if scope == "graph" else None))
        S.append(model.anomaly_score(h, hr).cpu().numpy())
        Y.append(b.y.cpu().numpy())
    return np.concatenate(S), np.concatenate(Y)


def anomaly_report(scores, y, quantile=None, val_scores=None):
    """Đánh giá chế độ phát hiện bất thường nhị phân.

    Ngưỡng lấy từ PHÂN VỊ của điểm số trên tập KIỂM ĐỊNH BÌNH THƯỜNG, không
    lấy từ tập kiểm thử — nếu dò ngưỡng trên tập kiểm thử thì kết quả đã dùng
    nhãn kiểm thử và không còn là đánh giá trung thực.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    yb = (y > 0).astype(int)
    out = {}
    if len(np.unique(yb)) > 1:
        out["roc_auc"] = float(roc_auc_score(yb, scores))
        out["pr_auc"] = float(average_precision_score(yb, scores))
    if val_scores is not None and quantile is not None:
        thr = float(np.quantile(val_scores, quantile))
        pred = (scores > thr).astype(int)
        out["threshold"] = thr
        out["binary"] = binary_metrics(yb, pred)
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints/GAT-Transformer")
    ap.add_argument("--stage", default="ssl", choices=["ssl", "finetune", "scratch"],
                    help="ssl = tiền huấn luyện tự giám sát trên nút bình thường; "
                         "finetune = tinh chỉnh có giám sát từ checkpoint SSL; "
                         "scratch = huấn luyện có giám sát KHÔNG tiền huấn luyện "
                         "(nhánh đối chứng để đo giá trị của giai đoạn 1).")
    ap.add_argument("--init-from", default=None,
                    help="Thư mục checkpoint giai đoạn SSL để khởi tạo.")
    ap.add_argument("--backbone", default="negat", choices=["negat", "edgegat"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--window", type=int, default=512,
                    help="Số nút mỗi cửa sổ tự chú ý.")
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--window-scope", default="batch", choices=["batch", "graph"],
                    help="Phạm vi gom nút vào MỘT cửa sổ tự chú ý. "
                         "batch = trên toàn lô (mặc định của bài báo gốc; một "
                         "cửa sổ có thể gom nút từ ~32 đồ thị khác nhau). "
                         "graph = chỉ gom nút TRONG CÙNG một đồ thị, tránh việc "
                         "Transformer học 'đồng xuất hiện' giữa những nút chưa "
                         "bao giờ cùng tồn tại trong một mạng thật.")
    # MẶC ĐỊNH ĐÃ ĐỔI 64 -> 32. Lý do ở docstring của MaskedAutoencoder:
    # d_model PHẢI nhỏ hơn hoặc bằng chiều embedding của GNN, nếu không
    # bộ tự mã hoá KHÔNG có nút thắt và sụp về hàm đồng nhất.
    ap.add_argument("--d-model", type=int, default=32,
                    help="Chiều ẩn Transformer. PHẢI <= chiều embedding GNN "
                         "(out_dim, mặc định 32) để có nút thắt thật.")
    ap.add_argument("--tr-layers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=64,
                    help="Chiều mạng truyền thẳng, khuyến nghị 2 x d_model.")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--no-stopgrad", action="store_true",
                    help="TÁI HIỆN LỖI CŨ: bỏ stop-gradient ở nhánh đích của "
                         "hàm mất mát SSL. Chỉ dùng để chạy đối chứng cho thấy "
                         "biểu diễn bị sụp (MSE -> 0 nhưng ROC-AUC không tăng). "
                         "KHÔNG dùng cho kết quả chính thức.")
    ap.add_argument("--freeze-gnn", action="store_true",
                    help="Giai đoạn 2: đóng băng bộ mã hoá đồ thị, chỉ huấn luyện "
                         "đầu phân loại. Dùng để đo riêng chất lượng biểu diễn "
                         "mà giai đoạn 1 học được.")
    ap.add_argument("--val-quantile", type=float, default=0.95,
                    help="Phân vị điểm số trên tập kiểm định để đặt ngưỡng bất thường.")
    a = ap.parse_args()

    set_seed(a.seed)
    cfg = load_config(a.config)
    cfg["model"]["ssl_backbone"] = a.backbone
    cfg.setdefault("transformer", {})
    cfg["transformer"].update(d_model=a.d_model, layers=a.tr_layers,
                              dim_ff=a.dim_ff, nhead=a.nhead, mask_ratio=a.mask_ratio)
    bs = a.batch_size or cfg["train"]["batch_size"]
    device = get_device(cfg["device"])
    os.makedirs(a.ckpt_dir, exist_ok=True)

    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    ld = lambda n: torch.load(os.path.join(a.data_dir, n), weights_only=False)
    train_g, val_g, test_g = ld("train.pt"), ld("val.pt"), ld("test.pt")
    log.info("Tải: train=%d val=%d test=%d | d_n=%d d_e=%d | %d lớp",
             len(train_g), len(val_g), len(test_g),
             meta["n_node_features"], meta["n_edge_features"], meta["num_classes"])

    from src.models.gat_transformer import build_gat_transformer
    model = build_gat_transformer(cfg, meta).to(device)

    if a.init_from:
        w = os.path.join(a.init_from, "best_model.pt")
        sd = torch.load(w, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("Khởi tạo từ %s | thiếu %d khoá, thừa %d khoá "
                 "(đầu phân loại chưa có ở giai đoạn SSL là bình thường)",
                 a.init_from, len(missing), len(unexpected))

    ck = os.path.join(a.ckpt_dir, "best_model.pt")
    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    T, Tw = a.epochs, t.get("warmup_epochs", 6)
    mr = t.get("min_lr_ratio", 0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / max(Tw, 1) if e < Tw
        else mr + (1 - mr) * 0.5 * (1 + np.cos(np.pi * (e - Tw) / max(T - Tw, 1))))
    gclip = t.get("grad_clip", 1.0)

    # ══════════════════════════════════════════════════════════
    if a.stage == "ssl":
        ben_tr, ben_va = filter_benign(train_g), filter_benign(val_g)
        if not ben_tr or not ben_va:
            return 1
        tr_ld = DataLoader(ben_tr, batch_size=bs, shuffle=True)
        va_ld = DataLoader(ben_va, batch_size=bs, shuffle=False)
        te_ld = DataLoader(test_g, batch_size=bs, shuffle=False)
        stopper = EarlyStopping(patience=t.get("early_stopping_patience", 30), mode="min")
        best = float("inf")

        for ep in range(1, a.epochs + 1):
            model.train(); tot = nb = 0
            for b in tr_ld:
                b = b.to(device)
                opt.zero_grad()
                h, hr = model(b.x, b.edge_index, b.edge_attr, mode="ssl", window=a.window,
                                 batch_vec=(b.batch if a.window_scope == "graph" else None))
                # CHẶN GRADIENT Ở NHÁNH ĐÍCH (stop-gradient).
                #
                # Nếu để mse_loss(hr, h) với h CÒN gradient, hàm mất mát có một
                # nghiệm tầm thường: bộ mã hoá tự CO NHỎ h về gần 0, khi đó bộ
                # giải mã khớp được ngay và MSE -> 0 mà KHÔNG học được gì. Đây
                # là hiện tượng SỤP BIỂU DIỄN (representation collapse), lý do
                # BYOL và SimSiam đều đặt stop-gradient ở nhánh đích.
                #
                # Dấu hiệu trong nhật ký khi bị sụp: MSE tụt xuống đúng 0.00000
                # và đứng yên hàng chục epoch, trong khi ROC-AUC của giai đoạn
                # phát hiện bất thường chỉ quanh 0,5-0,7.
                target = h if a.no_stopgrad else h.detach()
                loss = torch.nn.functional.mse_loss(hr, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
                opt.step()
                tot += float(loss.detach()); nb += 1
            sched.step()

            model.eval(); vt = vn = 0.0; vvar = 0.0
            with torch.no_grad():
                for b in va_ld:
                    b = b.to(device)
                    h, hr = model(b.x, b.edge_index, b.edge_attr, mode="ssl", window=a.window,
                                 batch_vec=(b.batch if a.window_scope == "graph" else None))
                    vt += float(torch.nn.functional.mse_loss(hr, h))
                    vvar += float(h.var()); vn += 1
            vl = vt / max(vn, 1)
            hvar = vvar / max(vn, 1)
            # MSE TƯƠNG ĐỐI = MSE / var(h). Chỉ số này BẤT BIẾN với phép co
            # giãn h, nên nó phân biệt được "tái tạo tốt" với "đã co nhỏ tín
            # hiệu". MSE tuyệt đối nhỏ mà MSE tương đối không nhỏ => đang sụp.
            vrel = vl / max(hvar, 1e-12)
            is_best = stopper.step(vl)
            if is_best:
                best = vl
                torch.save(model.state_dict(), ck)
            if ep % 5 == 0 or ep == 1 or is_best:
                log.info("Epoch %3d | MSE train %.5f | MSE val %.5f | "
                         "var(h) %.5f | MSE tương đối %.4f%s",
                         ep, tot / max(nb, 1), vl, hvar, vrel,
                         "  <-- best" if is_best else "")
            if hvar < 1e-6:
                log.warning("var(h) = %.3g — biểu diễn đã SỤP về hằng số. "
                            "Bộ mã hoá không còn mang thông tin. Kiểm tra lại "
                            "stop-gradient (--no-stopgrad có đang bật không?).", hvar)
            if stopper.should_stop:
                log.info("Dừng sớm tại epoch %d.", ep); break

        model.load_state_dict(torch.load(ck, map_location=device))
        vs, _ = collect_scores(model, va_ld, device, a.window, a.window_scope)
        ts, ty = collect_scores(model, te_ld, device, a.window, a.window_scope)
        rep = anomaly_report(ts, ty, quantile=a.val_quantile, val_scores=vs)
        log.info("== GIAI ĐOẠN 1 — PHÁT HIỆN BẤT THƯỜNG (không dùng nhãn khi huấn luyện) ==")
        if "roc_auc" in rep:
            log.info("  ROC-AUC = %.4f | PR-AUC = %.4f", rep["roc_auc"], rep["pr_auc"])
        if "binary" in rep:
            log.info("  Ngưỡng = phân vị %.2f điểm số trên tập kiểm định bình thường "
                     "(%.4f)", a.val_quantile, rep["threshold"])
            log.info("\n%s", format_binary(rep["binary"],
                     "PHÁT HIỆN BẤT THƯỜNG — ngưỡng từ tập KIỂM ĐỊNH"))
        log.info("MSE kiểm định tốt nhất = %.5f", best)
        json.dump({k: v for k, v in rep.items() if k != "binary"} |
                  {"binary": rep.get("binary", {}), "best_val_mse": best},
                  open(os.path.join(a.ckpt_dir, "ssl_report.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)

    # ══════════════════════════════════════════════════════════
    else:
        if a.freeze_gnn:
            for p in model.gnn.parameters():
                p.requires_grad = False
            log.info("ĐÃ ĐÓNG BĂNG bộ mã hoá đồ thị — chỉ huấn luyện Transformer "
                     "và đầu phân loại.")
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=t["lr"], weight_decay=t["weight_decay"])
        tr_ld = DataLoader(train_g, batch_size=bs, shuffle=True)
        va_ld = DataLoader(val_g, batch_size=bs, shuffle=False)
        te_ld = DataLoader(test_g, batch_size=bs, shuffle=False)
        crit = make_criterion(cfg, train_g, meta["num_classes"], device)

        use_ema = t.get("use_ema", True)
        ema = EMA(model, decay=t.get("ema_decay", 0.995)) if use_ema else None
        eval_model = copy.deepcopy(model) if ema is not None else model
        stopper = EarlyStopping(patience=t.get("early_stopping_patience", 30), mode="max")
        xi = t.get("val_smooth_beta", 0.6); sm = None; best = -1.0

        class _W(torch.nn.Module):
            """Bọc mô hình để evaluate() gọi được với chữ ký chuẩn."""
            def __init__(self, m, win): super().__init__(); self.m = m; self.win = win
            def forward(self, x, ei, ea=None, **kw):
                return self.m(x, ei, ea, mode="clf", window=self.win)

        for ep in range(1, a.epochs + 1):
            model.train(); tot = nb = 0
            for b in tr_ld:
                b = b.to(device)
                opt.zero_grad()
                logits = model(b.x, b.edge_index, b.edge_attr, mode="clf", window=a.window)
                loss = crit(logits, b.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
                opt.step()
                if ema is not None:
                    ema.update(model)
                tot += float(loss); nb += 1
            sched.step()

            if ema is not None:
                ema.copy_to(eval_model)
            vm = evaluate(_W(eval_model if ema is not None else model, a.window),
                          va_ld, device, meta["num_classes"], meta["class_names"])
            f1 = vm["macro_f1"]
            sm = f1 if sm is None else xi * sm + (1 - xi) * f1
            is_best = stopper.step(sm)
            if is_best:
                best = sm
                torch.save((eval_model if ema is not None else model).state_dict(), ck)
            if ep % 5 == 0 or ep == 1 or is_best:
                log.info("Epoch %3d | loss %.4f | val %.4f (mượt %.4f)%s",
                         ep, tot / max(nb, 1), f1, sm, "  <-- best" if is_best else "")
            if stopper.should_stop:
                log.info("Dừng sớm tại epoch %d.", ep); break

        model.load_state_dict(torch.load(ck, map_location=device))
        tm = evaluate(_W(model, a.window), te_ld, device,
                      meta["num_classes"], meta["class_names"], with_binary=True)
        log.info("== KẾT QUẢ TEST (GAT-Transformer IDS, %s) ==\n%s",
                 a.stage, format_report(tm, meta["class_names"]))
        log.info("\n%s", format_binary(tm["binary"]))
        log.info("Best val macro-F1 (đã làm mượt) = %.4f", best)

        # Lưu ĐẦY ĐỦ mọi siêu tham số cần để dựng lại mô hình. Trước đây bản
        # ghi này thiếu edge_latent, node_heads, num_layers, dropout, nhead,
        # clf_hidden... nên test_all.py không thể tái dựng đúng kiến trúc.
        m_ = cfg["model"]
        mc = {"in_dim": meta["n_node_features"], "edge_dim": meta["n_edge_features"],
              "num_classes": meta["num_classes"], "model_type": "GAT-Transformer",
              "backbone": a.backbone, "stage": a.stage,
              "hidden_dim": m_["hidden_dim"], "out_dim": m_["out_dim"],
              "edge_latent": m_.get("edge_latent", 24),
              "node_heads": m_.get("node_heads", 2),
              "heads": m_.get("heads", 3),
              "num_layers": m_.get("num_layers", 3),
              "dropout": m_.get("dropout", 0.3),
              "edge_score_mode": m_.get("edge_score_mode", "full"),
              "edge_agg": m_.get("edge_agg", "mean_deg"),
              "edge_update": m_.get("edge_update", True),
              "use_edge_features": m_.get("use_edge_features", True),
              "clf_hidden": m_.get("clf_hidden", 128),
              "d_model": a.d_model, "nhead": a.nhead,
              "tr_layers": a.tr_layers, "dim_ff": a.dim_ff,
              "mask_ratio": a.mask_ratio, "window": a.window,
              "window_scope": a.window_scope,
              "init_from": a.init_from, "freeze_gnn": bool(a.freeze_gnn),
              "stopgrad": not a.no_stopgrad,
              "class_names": meta["class_names"]}
        json.dump(mc, open(os.path.join(a.ckpt_dir, "model_config.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    log.info("Đã lưu -> %s | %d tham số", a.ckpt_dir, model.count_parameters())
    return 0


if __name__ == "__main__":
    sys.exit(main())
