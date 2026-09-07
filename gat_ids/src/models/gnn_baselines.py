"""Baseline GNN: GCN, Edge-GraphSAGE, Deep MLP cho node classification.

Tất cả dùng cùng interface forward(x, edge_index, edge_attr) để tương thích
với training loop và metrics của framework. Khác nhau ở cách xử lý cấu trúc:
  - MLP: BỎ QUA topology hoàn toàn (chỉ dùng node features)
  - GCN: dùng topology nhưng KHÔNG dùng edge features
  - Edge-GraphSAGE: dùng CẢ topology VÀ edge features (như E-GraphSAGE paper)
  - (Edge-aware GAT: dùng topology + edge features + attention → đề xuất chính)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops

from .edge_gat import DyT


# ════════════════════════════════════════════════════════════════
#  Deep MLP — baseline không dùng đồ thị
# ════════════════════════════════════════════════════════════════
class DeepMLP(nn.Module):
    """MLP 3 lớp ẩn. Hoàn toàn bỏ qua cấu trúc đồ thị — chỉ dùng node features.
    Nếu MLP đạt bằng GNN → cấu trúc đồ thị không giúp gì."""

    def __init__(self, in_dim, edge_dim, hidden_dim=64, out_dim=32,
                 num_classes=6, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), DyT(hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), DyT(hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), DyT(out_dim), nn.ELU(),
        )
        self.classifier = nn.Linear(out_dim, num_classes)
        self._last_attn = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False,
                return_feat=False):
        h = self.net(x)
        logits = self.classifier(h)
        if return_feat:          # biểu diễn TRƯỚC bộ phân loại, giữ đồ thị tính toán
            return logits, h
        if return_attention:
            # MLP không có attention — trả dummy
            n = x.shape[0]
            dummy_ei = torch.stack([torch.arange(n, device=x.device),
                                    torch.arange(n, device=x.device)])
            dummy_a = torch.ones(n, device=x.device) * 0.5
            self._last_attn = (dummy_ei, dummy_a)
            return logits, self._last_attn
        return logits

    @torch.no_grad()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════
#  GCN — dùng topology, KHÔNG dùng edge features
# ════════════════════════════════════════════════════════════════
class GCNBaseline(nn.Module):
    """2-layer GCN (Kipf & Welling 2017). Dùng cấu trúc đồ thị nhưng
    KHÔNG dùng edge features — để đo giá trị riêng của topology."""

    def __init__(self, in_dim, edge_dim, hidden_dim=64, out_dim=32,
                 num_classes=6, dropout=0.3, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.dyt1 = DyT(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dyt2 = DyT(out_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(out_dim, num_classes)
        self._last_attn = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False,
                return_feat=False):
        x1 = F.elu(self.dyt1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.elu(self.dyt2(self.conv2(x1, edge_index)))
        logits = self.classifier(x2)
        if return_feat:
            return logits, x2
        if return_attention:
            dummy_a = torch.ones(edge_index.shape[1], device=x.device) * 0.5
            self._last_attn = (edge_index, dummy_a)
            return logits, self._last_attn
        return logits

    @torch.no_grad()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════
#  Edge-GraphSAGE — dùng topology VÀ edge features (E-GraphSAGE)
# ════════════════════════════════════════════════════════════════
class EdgeSAGEConv(MessagePassing):
    """GraphSAGE conv tích hợp edge features vào message (theo E-GraphSAGE).

    Message: m_ij = W · [h_j || edge_ij]  (concatenate node + edge features)
    Aggregate: mean
    Update: h_i' = W2 · [h_i || agg(m_ij)]
    """
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr='mean')
        self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.lin_upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: source node features [E, in_dim]
        # edge_attr: [E, edge_dim]
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.lin_upd(torch.cat([x, aggr_out], dim=-1))


class EdgeGraphSAGE(nn.Module):
    """2-layer Edge-GraphSAGE (Lo et al. 2022). Dùng CẢ topology VÀ edge features
    nhưng KHÔNG có attention — để đo giá trị riêng của attention mechanism."""

    def __init__(self, in_dim, edge_dim, hidden_dim=64, out_dim=32,
                 num_classes=6, dropout=0.3, **kwargs):
        super().__init__()
        self.conv1 = EdgeSAGEConv(in_dim, edge_dim, hidden_dim)
        self.dyt1 = DyT(hidden_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, edge_dim, out_dim)
        self.dyt2 = DyT(out_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(out_dim, num_classes)
        self._last_attn = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False,
                return_feat=False):
        ea = edge_attr if edge_attr is not None else torch.zeros(
            edge_index.shape[1], 1, device=x.device)
        x1 = F.elu(self.dyt1(self.conv1(x, edge_index, ea)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.elu(self.dyt2(self.conv2(x1, edge_index, ea)))
        logits = self.classifier(x2)
        if return_feat:
            return logits, x2
        if return_attention:
            dummy_a = torch.ones(edge_index.shape[1], device=x.device) * 0.5
            self._last_attn = (edge_index, dummy_a)
            return logits, self._last_attn
        return logits

    @torch.no_grad()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════
#  Factory — dựng model theo tên
# ════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "mlp": DeepMLP,
    "gcn": GCNBaseline,
    "edge_graphsage": EdgeGraphSAGE,
}


def build_baseline(name: str, meta: dict, cfg: dict):
    """Tạo baseline model theo tên."""
    m = cfg["model"]
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Baseline '{name}' không tồn tại. Có: {list(MODEL_REGISTRY)}")
    return cls(
        in_dim=meta["n_node_features"],
        edge_dim=meta["n_edge_features"],
        hidden_dim=m["hidden_dim"],
        out_dim=m["out_dim"],
        num_classes=meta["num_classes"],
        dropout=m["dropout"],
    )
