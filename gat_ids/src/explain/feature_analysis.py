"""Phân tích MODEL HỌC ĐẶC TRƯNG NHƯ THẾ NÀO — vượt ra ngoài accuracy/F1.

Bốn góc nhìn để biết model học ĐÚNG (học tín hiệu thật) chứ không học vẹt:

  1. Embedding separability: chiếu embedding tầng cuối xuống 2D (t-SNE/PCA),
     đo độ tách lớp bằng silhouette score. Lớp tách rõ = model học được
     biểu diễn có ý nghĩa, không phải nhớ máy móc.

  2. Permutation feature importance: hoán vị (xáo) từng feature rồi đo F1 sụt
     bao nhiêu. Feature càng quan trọng -> sụt càng nhiều. Cho biết model
     DỰA VÀO đặc trưng nào -> kiểm tra có khớp domain knowledge RPL không.

  3. Attention analysis: thống kê trọng số attention trên cạnh quanh node
     sinkhole vs normal. Nếu model "chú ý" đúng cạnh bất thường = học đúng.

  4. Confidence calibration: phân bố xác suất dự đoán. Model học đúng thì
     tự tin ở mẫu dễ, do dự ở mẫu khó — không "tự tin mù quáng".
"""
from __future__ import annotations
import numpy as np
import torch

from ..utils.metrics import collect_predictions, compute_metrics
from ..utils.common import get_logger

log = get_logger()


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Lấy embedding tầng áp chót (trước classifier) + nhãn."""
    model.eval()
    embs, ys = [], []

    # Hook vào input của classifier để lấy embedding
    captured = {}
    def hook(module, inp, out):
        captured["emb"] = inp[0].detach()
    h = model.classifier.register_forward_hook(hook)

    for batch in loader:
        batch = batch.to(device)
        _ = model(batch.x, batch.edge_index, batch.edge_attr)
        embs.append(captured["emb"].cpu())
        ys.append(batch.y.cpu())
    h.remove()
    return torch.cat(embs).numpy(), torch.cat(ys).numpy()


def embedding_separability(model, loader, device):
    """Silhouette score trên embedding — đo độ tách lớp trong không gian học được."""
    emb, y = extract_embeddings(model, loader, device)
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(y)) < 2:
            return {"silhouette": None, "note": "chỉ 1 lớp"}
        # subsample cho nhanh nếu lớn
        if len(y) > 3000:
            idx = np.random.default_rng(0).choice(len(y), 3000, replace=False)
            emb, y = emb[idx], y[idx]
        sil = float(silhouette_score(emb, y))
        return {"silhouette": round(sil, 4),
                "interpret": _interp_sil(sil)}
    except ImportError:
        # tự tính khoảng cách inter/intra class nếu thiếu sklearn
        return _manual_separability(emb, y)


def _interp_sil(s):
    if s > 0.5: return "tách lớp RẤT RÕ (model học biểu diễn tốt)"
    if s > 0.25: return "tách lớp khá (chấp nhận được)"
    if s > 0: return "tách lớp yếu (cần xem lại features)"
    return "KHÔNG tách lớp (model có thể học vẹt/nhiễu)"


def _manual_separability(emb, y):
    classes = np.unique(y)
    centroids = {c: emb[y == c].mean(0) for c in classes}
    intra = np.mean([np.linalg.norm(emb[i] - centroids[y[i]]) for i in range(len(y))])
    inter = np.mean([np.linalg.norm(centroids[a] - centroids[b])
                     for a in classes for b in classes if a < b])
    ratio = inter / (intra + 1e-9)
    return {"inter_intra_ratio": round(float(ratio), 3),
            "interpret": "càng lớn càng tách lớp tốt (>2 là tốt)"}


def permutation_importance(model, loader, device, num_classes, class_names,
                           n_node_features):
    """Đo tầm quan trọng từng NODE feature bằng cách hoán vị giá trị của nó.

    Trả về list (feature_idx, f1_drop) sắp giảm dần.
    """
    # F1 gốc
    base = compute_metrics(*collect_predictions(model, loader, device),
                           num_classes, class_names)["macro_f1"]

    # Gom toàn bộ batch về 1 lần để hoán vị nhất quán
    all_graphs = []
    for batch in loader:
        all_graphs.append(batch)

    drops = []
    rng = np.random.default_rng(0)
    for f in range(n_node_features):
        # Hoán vị feature f trên từng graph
        saved = []
        for b in all_graphs:
            col = b.x[:, f].clone()
            saved.append(col)
            perm = torch.randperm(b.x.shape[0])
            b.x[:, f] = col[perm]
        # đo lại
        m = compute_metrics(*collect_predictions(model, loader, device),
                            num_classes, class_names)["macro_f1"]
        drops.append((f, round(base - m, 4)))
        # khôi phục
        for b, col in zip(all_graphs, saved):
            b.x[:, f] = col

    drops.sort(key=lambda x: -x[1])
    return base, drops


@torch.no_grad()
def attention_on_attack_nodes(model, loader, device):
    """So sánh attention trung bình trên cạnh nối node attack vs node normal.

    Nếu model học đúng, attention quanh node attack nên KHÁC biệt rõ.
    """
    model.eval()
    atk_attn, norm_attn = [], []
    for batch in loader:
        batch = batch.to(device)
        _, (ei, alpha) = model(batch.x, batch.edge_index,
                               batch.edge_attr, return_attention=True)
        if alpha.dim() > 1:
            alpha = alpha.mean(1)
        y = batch.y
        for e in range(ei.shape[1]):
            dst = ei[1, e].item()
            if dst < y.numel():
                (atk_attn if int(y[dst]) == 1 else norm_attn).append(float(alpha[e]))
    res = {
        "attn_mean_on_attack_nodes": round(float(np.mean(atk_attn)), 4) if atk_attn else None,
        "attn_mean_on_normal_nodes": round(float(np.mean(norm_attn)), 4) if norm_attn else None,
    }
    if atk_attn and norm_attn:
        res["attention_discriminative"] = abs(
            np.mean(atk_attn) - np.mean(norm_attn)) > 0.01
    return res


def run_feature_analysis(model, loader, device, meta):
    """Chạy cả 4 phân tích, in báo cáo tổng hợp."""
    log.info("=== PHÂN TÍCH MODEL HỌC ĐẶC TRƯNG ===")

    sep = embedding_separability(model, loader, device)
    log.info("[1] Embedding separability: %s", sep)

    base, drops = permutation_importance(
        model, loader, device, meta["num_classes"],
        meta["class_names"], meta["n_node_features"])
    log.info("[2] Permutation importance (top-5 node features quan trọng nhất):")
    for f, d in drops[:5]:
        log.info("      node_feature[%d] -> F1 sụt %.4f khi xáo", f, d)

    attn = attention_on_attack_nodes(model, loader, device)
    log.info("[3] Attention trên node attack vs normal: %s", attn)

    return {"separability": sep, "perm_importance_top5": drops[:5],
            "attention": attn, "base_macro_f1": base}
