"""Engine huấn luyện dùng chung (centralized + federated local) — BẢN ỔN ĐỊNH.

Sửa trực tiếp bệnh trong log 100 epochs: val macro-F1 dao động +-0.08 mỗi epoch,
"best" checkpoint chỉ là 1 đỉnh nhiễu may mắn (0.8703@ep98 -> test 0.8634).

Bốn cơ chế ổn định thêm vào (đều có công tắc, mặc định BẬT):
  * AdamW + weight-decay tách rời (tốt hơn Adam cho regularize).
  * Cosine LR + warmup: LR khởi động nhỏ rồi giảm mượt -> hết dao động.
  * Gradient clipping (max_norm): chặn bước cập nhật nổ.
  * EMA (Exponential Moving Average) trọng số: đánh giá & lưu bằng trọng số
    trung bình trượt -> đường val cực mượt, checkpoint không còn ăn may.
  * Chọn checkpoint theo val macro-F1 ĐÃ LÀM MƯỢT (moving average) thay vì
    giá trị tức thời -> lấy model ổn định thật, không lấy đỉnh nhiễu.
"""
from __future__ import annotations
import os
import copy
import math
import torch
import torch.nn as nn

from .metrics import collect_predictions, compute_metrics, format_report
from .common import EarlyStopping, get_logger

log = get_logger()


# ──────────────────────────────────────────────────────────────
#  EMA trọng số
# ──────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    fedprox_mu=0.0, global_state=None, grad_clip=1.0, ema=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)
        if fedprox_mu > 0 and global_state is not None:
            prox = 0.0
            for name, param in model.named_parameters():
                gp = global_state[name].to(device)
                prox = prox + ((param - gp) ** 2).sum()
            loss = loss + (fedprox_mu / 2.0) * prox
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names, with_binary=False):
    """with_binary=True: bổ sung khoá 'binary' chứa bộ chỉ số nhị phân.

    Hai chế độ trả lời hai câu hỏi khác nhau và không thay thế được nhau:
    đa lớp hỏi 'đây là loại tấn công nào', nhị phân hỏi 'có tấn công không'.
    """
    y_true, y_pred = collect_predictions(model, loader, device)
    m = compute_metrics(y_true, y_pred, num_classes, class_names)
    if with_binary:
        from .binary_eval import binary_metrics
        m["binary"] = binary_metrics(y_true, y_pred)
    return m


# ──────────────────────────────────────────────────────────────
#  Tạo criterion — hỗ trợ cả loss cũ và các loss long-tail mới
# ──────────────────────────────────────────────────────────────
def _node_counts(train_graphs, num_classes):
    counts = torch.zeros(num_classes)
    for g in train_graphs:
        for c in g.y:
            counts[int(c)] += 1
    return counts


def make_criterion(cfg, train_graphs, num_classes, device, orig_counts=None):
    """orig_counts: phân bố lớp TRƯỚC khi resample.

    SỬA LỖI QUAN TRỌNG: trước đây trọng số class-balanced được tính trên tập
    ĐÃ qua SMOTE, khi đó lớp hiếm đã bị nâng lên ~7000 mẫu nên CB-Focal tưởng
    dữ liệu đã cân bằng. Đo được: tỉ lệ w(hiếm)/w(Normal) tụt từ 18 xuống 2 —
    SMOTE và CB-Focal TRIỆT TIÊU NHAU. Truyền orig_counts để trọng số phản ánh
    độ hiếm THẬT của dữ liệu gốc.
    """
    loss_type = cfg["train"].get("loss", "cb_focal")
    counts = (orig_counts.to(device) if orig_counts is not None
              else _node_counts(train_graphs, num_classes).to(device))

    # loss long-tail mới (Cui/Cao/Menon)
    if loss_type in ("cb_focal", "class_balanced_focal", "ldam",
                     "logit_adj", "logit_adjustment"):
        from .losses import build_loss
        log.info("Loss long-tail: %s | counts(min=%d,max=%d)",
                 loss_type, int(counts.min()), int(counts.max()))
        return build_loss(loss_type, counts.cpu(), cfg["train"]).to(device)

    # tương thích ngược: focal / weighted_ce cũ
    weight = None
    if cfg["train"].get("class_weighted_loss", False):
        from ..data.loader import compute_class_weights
        weight = compute_class_weights(train_graphs, num_classes).to(device)
    if loss_type == "focal":
        from .losses import FocalLoss
        gamma = cfg["train"].get("focal_gamma", 1.0)
        log.info("Focal Loss (gamma=%.1f)", gamma)
        return FocalLoss(alpha=weight, gamma=gamma)
    log.info("%s CrossEntropy", "Weighted" if weight is not None else "")
    return nn.CrossEntropyLoss(weight=weight)


# ──────────────────────────────────────────────────────────────
#  Vòng huấn luyện centralized ổn định
# ──────────────────────────────────────────────────────────────
def train_centralized(model, loaders, criterion, cfg, meta, device):
    train_loader, val_loader, test_loader = loaders
    t = cfg["train"]
    lr = t.get("lr", 0.002)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=t.get("weight_decay", 5e-4))

    epochs = t["epochs"]
    warmup = t.get("warmup_epochs", max(3, epochs // 20))
    min_lr_ratio = t.get("min_lr_ratio", 0.05)

    def lr_lambda(ep):                                    # warmup + cosine
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_ema = t.get("use_ema", True)
    ema = EMA(model, decay=t.get("ema_decay", 0.995)) if use_ema else None
    grad_clip = t.get("grad_clip", 1.0)

    stopper = EarlyStopping(patience=t["early_stopping_patience"], mode="max")
    os.makedirs(t["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(t["ckpt_dir"], "best_centralized.pt")

    best_smooth, ema_val = -1.0, None
    smooth_beta = t.get("val_smooth_beta", 0.6)
    eval_model = copy.deepcopy(model)                     # dùng để eval bằng EMA

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion,
                               device, grad_clip=grad_clip, ema=ema)
        scheduler.step()

        # đánh giá trên trọng số EMA (mượt) nếu bật
        if ema is not None:
            ema.copy_to(eval_model)
            val_m = evaluate(eval_model, val_loader, device,
                             meta["num_classes"], meta["class_names"])
        else:
            val_m = evaluate(model, val_loader, device,
                             meta["num_classes"], meta["class_names"])

        # làm mượt val macro-F1 để chọn checkpoint (chống ăn đỉnh nhiễu)
        v = val_m["macro_f1"]
        ema_val = v if ema_val is None else smooth_beta * ema_val + (1 - smooth_beta) * v
        is_best = stopper.step(ema_val)
        if is_best:
            best_smooth = ema_val
            save_state = (eval_model if ema is not None else model).state_dict()
            torch.save(save_state, ckpt_path)
        if epoch % 5 == 0 or is_best:
            log.info("Epoch %3d | loss %.4f | val %.4f (mượt %.4f) | lr %.2e%s",
                     epoch, loss, v, ema_val, scheduler.get_last_lr()[0],
                     "  <-- best" if is_best else "")
        if stopper.should_stop:
            log.info("Early stop @ epoch %d (best mượt val=%.4f)", epoch, best_smooth)
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = evaluate(model, test_loader, device,
                      meta["num_classes"], meta["class_names"])
    log.info("== KẾT QUẢ TEST (centralized) ==\n%s",
             format_report(test_m, meta["class_names"]))
    return model, test_m
