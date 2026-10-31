"""Giải thích cảnh báo: (1) attention weights sẵn có của GAT,
(2) GNNExplainer cho subgraph. Chạy ở pha inference (model đã train)."""
from __future__ import annotations
import torch
from ..utils.common import get_logger

log = get_logger()


@torch.no_grad()
def explain_by_attention(model, data, device, top_k=8):
    """Lấy các cạnh có attention cao nhất cho 1 graph — giải thích 'rẻ' nhất."""
    model.eval()
    data = data.to(device)
    logits, (edge_index, alpha) = model(
        data.x, data.edge_index, data.edge_attr, return_attention=True)
    # alpha shape [num_edges, heads] -> trung bình các head
    if alpha.dim() > 1:
        alpha = alpha.mean(dim=1)
    k = min(top_k, alpha.shape[0])
    top_val, top_idx = torch.topk(alpha, k)
    edges = []
    for v, i in zip(top_val.tolist(), top_idx.tolist()):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        edges.append({"edge": (s, d), "attention": round(v, 4)})
    preds = logits.argmax(dim=1).cpu().tolist()
    return {"top_edges": edges, "node_predictions": preds}


def explain_by_gnnexplainer(model, data, device, epochs=100):
    """Dùng GNNExplainer (PyG) để rút edge mask + feature mask.

    Bao trong try/except vì API GNNExplainer thay đổi giữa các phiên bản PyG.
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        model.eval()
        data = data.to(device)
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification",
                              task_level="node", return_type="raw"),
        )
        # giải thích nút 0 làm ví dụ
        explanation = explainer(data.x, data.edge_index,
                                edge_attr=data.edge_attr, index=0)
        em = explanation.edge_mask.detach().cpu()
        return {"edge_mask_top": torch.topk(em, min(8, em.numel())).indices.tolist(),
                "edge_mask_mean": float(em.mean())}
    except Exception as e:                       # noqa: BLE001
        log.warning("GNNExplainer bỏ qua (API/phiên bản): %s", e)
        return {"note": "GNNExplainer không chạy được ở môi trường này",
                "error": str(e)}


def run_explanations(model, test_graphs, device, cfg):
    if not cfg["explain"]["enabled"] or len(test_graphs) == 0:
        return {}
    sample = test_graphs[0]
    attn = explain_by_attention(model, sample, device,
                                cfg["explain"]["top_k_edges"])
    gnnx = explain_by_gnnexplainer(model, sample, device,
                                   cfg["explain"]["gnnexplainer_epochs"])
    log.info("XAI attention top-edges (graph mẫu): %s", attn["top_edges"][:3])
    return {"attention": attn, "gnnexplainer": gnnx}
