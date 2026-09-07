"""Hàm mất mát cho mất cân bằng lớp CỰC ĐOAN (Normal ~87% node, dis ~0.15%).

Bài báo Amino-Acid IDS xử lý mất cân bằng bằng SMOTE (resampling). Framework
của bạn đã có GraphSMOTE-lite; ở tầng LOSS ta bổ sung 3 lựa chọn hiện đại hơn
Focal thuần, giải quyết đúng bệnh trong log 100 epochs:

    Triệu chứng: train loss chạm sàn 0.048 rất sớm (epoch ~12) rồi đứng yên,
    val macro-F1 dao động ±0.08. Nghĩa là loss bị Normal chi phối; gradient từ
    lớp hiếm quá nhỏ -> lớp như worst_parent (F1 0.41 -> 0.15) bị "bỏ rơi".

Ba vũ khí (chọn 1 qua config train.loss):
  1. cb_focal   — Class-Balanced Focal (Cui et al., CVPR 2019). Trọng số theo
                  "số mẫu hiệu dụng" (1-beta^n)/(1-beta) ON HOA hơn nghịch tần
                  suất (numpy: tỉ lệ hiếm/Normal 91x thay vì 573x) -> ít
                  over-correct precision như lỗi Focal+class_weight bạn từng gặp.
  2. ldam       — Label-Distribution-Aware Margin (Cao et al., NeurIPS 2019).
                  Đẩy biên quyết định RỘNG HƠN cho lớp hiếm: margin_c ~ n_c^(-1/4).
                  Hợp khi ranh giới lớp hiếm mờ (rank/sybil hay nhầm nhau).
  3. logit_adj  — Logit Adjustment (Menon et al., ICLR 2021). Cộng tau*log(pi_c)
                  vào logit lúc train; nền tảng Bayes cho long-tail, gần như
                  "free", không cần resample.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss (giữ nguyên - tương thích ngược)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Class-Balanced weights (Cui et al. 2019)
def class_balanced_weights(counts: torch.Tensor, beta: float = 0.9999,
                           cap: float = 5.0) -> torch.Tensor:
    """Trọng số class-balanced CÓ CHUẨN HOÁ TRẦN.

        E_c = (1 − β^{n_c}) / (1 − β)          [số mẫu hiệu dụng, Cui 2019]
        w_c = 1 / E_c
        w   = clip(w, w_min, cap · w_min)      [CHẶN TRẦN — mới]
        w   = w / mean(w)                       [chuẩn hoá trung bình = 1]

    VÌ SAO CẦN TRẦN: trọng số thô cách biệt quá xa. Trên phân bố thật của RADAR
    với β=0,9999, tỉ lệ w(hiếm)/w(Normal) = 18 lần. Hệ quả quan sát được ở lần
    chạy 120 epoch: mô hình BÙ QUÁ ĐÀ — delayed_reply đạt recall 0,978 nhưng
    precision chỉ 0,391, tức nó hút nhầm 408 node của lớp khác vào.

    cap là HẰNG SỐ THAM CHIẾU rõ ràng và ổn định: tỉ lệ tối đa cho phép giữa
    lớp được ưu tiên nhất và lớp ít nhất. Khác với các phương pháp thích nghi
    trực tuyến, nó KHÔNG phụ thuộc thang giá trị của hàm mất mát (vốn thay đổi
    trong quá trình huấn luyện: 0,079 ở epoch 1 xuống 0,0129 ở epoch 120).

        cap = 3   -> tỉ lệ 3,0   (cân bằng nhất, ưu tiên accuracy)
        cap = 5   -> tỉ lệ 5,0   (KHUYẾN NGHỊ, trung dung)
        cap = 10  -> tỉ lệ 10,0
        cap = 0   -> tắt trần, về hành vi cũ (tỉ lệ 18,0)
    """
    counts = counts.clamp_min(1.0).double()
    eff = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.double), counts)) / (1.0 - beta)
    w = 1.0 / eff
    if cap and cap > 0:
        lo = w.min()
        w = torch.clamp(w, min=lo, max=lo * float(cap))
    w = w / w.mean()
    return w.float()


class ClassBalancedFocalLoss(nn.Module):
    """Focal + trọng số class-balanced + LABEL SMOOTHING.

    Label smoothing bổ sung theo kết quả của "A Study on a NIDS Based on the
    Fusion of SAGEConv-GNN and a Transformer Encoder" (MDPI Electronics 2026):
    nhóm tác giả dùng class-weighted cross-entropy KÈM label smoothing để vừa
    xử lý mất cân bằng vừa cải thiện tổng quát hoá, cùng gradient clipping và
    early stopping theo macro-F1.

    VÌ SAO HỢP VỚI TA: 5 lớp họ-rank (rank/sinkhole/continuous_sinkhole/
    clone_id/sybil) có chữ ký GẦN GIỐNG NHAU nên hay nhầm lẫn chéo. Nhãn cứng
    (one-hot) ép mô hình cực kỳ tự tin vào một lớp, khuếch đại lỗi khi ranh
    giới mờ. Label smoothing làm mềm mục tiêu, giảm quá tự tin và thường nâng
    macro-F1 ở đúng nhóm lớp dễ nhầm này.
    """

    def __init__(self, counts: torch.Tensor, beta: float = 0.9999,
                 gamma: float = 1.0, label_smoothing: float = 0.0,
                 reduction: str = "mean", cap: float = 5.0):
        super().__init__()
        self.register_buffer("weight", class_balanced_weights(counts, beta, cap))
        self.gamma = gamma
        self.ls = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        w = self.weight.to(logits.device)
        if self.ls > 0:
            C = logits.size(1)
            # nhãn mềm: (1-ls) cho lớp đúng, ls/(C-1) chia đều cho lớp còn lại
            with torch.no_grad():
                true_dist = torch.full_like(logp, self.ls / (C - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.ls)
            ce = -(true_dist * logp).sum(dim=1) * w[target]
        else:
            ce = F.nll_loss(logp, target, weight=w, reduction="none")
        pt = torch.exp(-ce.clamp(max=20))
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# LDAM (Cao et al. 2019)
class LDAMLoss(nn.Module):
    """margin_c = C * n_c^{-1/4}. Lớp hiếm -> biên lớn hơn -> tổng quát tốt hơn."""

    def __init__(self, counts: torch.Tensor, max_m: float = 0.5,
                 weight: torch.Tensor | None = None, scale: float = 30.0):
        super().__init__()
        m = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0).float()))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", m)
        self.scale = scale
        self.weight = weight

    def forward(self, logits, target):
        idx = F.one_hot(target, num_classes=logits.size(1)).float()
        margins = (idx * self.m_list.to(logits.device)).sum(1, keepdim=True)
        logits_m = logits - margins
        out = torch.where(idx.bool(), logits_m, logits)
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(self.scale * out, target, weight=w)


# Logit Adjustment (Menon et al. 2021)
class LogitAdjustedLoss(nn.Module):
    """CE trên (logit + tau*log pi_c). Bù long-tail theo lý thuyết Bayes."""

    def __init__(self, counts: torch.Tensor, tau: float = 1.0):
        super().__init__()
        prior = counts.clamp_min(1.0).float()
        prior = prior / prior.sum()
        self.register_buffer("adj", tau * torch.log(prior + 1e-12))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.adj.to(logits.device), target)


def build_loss(name: str, counts: torch.Tensor, cfg_train: dict):
    """counts: tensor [num_classes] số node mỗi lớp trên TRAIN."""
    name = (name or "focal").lower()
    gamma = cfg_train.get("focal_gamma", 1.0)
    if name in ("cb_focal", "class_balanced_focal"):
        beta = cfg_train.get("cb_beta", 0.9999)
        ls = cfg_train.get("label_smoothing", 0.0)
        cap = cfg_train.get("weight_cap", 5.0)
        return ClassBalancedFocalLoss(counts, beta=beta, gamma=gamma,
                                      label_smoothing=ls, cap=cap)
    if name == "ldam":
        w = class_balanced_weights(counts, cfg_train.get("cb_beta", 0.9999),
                                   cfg_train.get("weight_cap", 5.0)) \
            if cfg_train.get("ldam_drw", True) else None
        return LDAMLoss(counts, max_m=cfg_train.get("ldam_max_m", 0.5), weight=w)
    if name in ("logit_adj", "logit_adjustment"):
        return LogitAdjustedLoss(counts, tau=cfg_train.get("logit_tau", 1.0))
    if name == "focal":
        return FocalLoss(gamma=gamma)
    return nn.CrossEntropyLoss()
