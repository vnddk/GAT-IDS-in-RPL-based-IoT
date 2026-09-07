# -*- coding: utf-8 -*-
"""
ne_gat.py — NE-GAT-IDS: kiến trúc hai kênh Node / Edge
======================================================

Cài đặt theo tài liệu thiết kế NE-GAT-IDS, KÈM BỐN SỬA LỖI so với bản gốc.
Mỗi sửa lỗi được ghi rõ lý do ngay tại chỗ, và đều có cờ cấu hình để bật/tắt
nhằm phục vụ ablation.

Ý tưởng gốc (giữ nguyên): thay vì trộn đặc trưng cạnh vào cùng một phép biến
đổi chú ý với đặc trưng nút, tách thành hai kênh học riêng rồi mới hợp nhất:

    h_i^N = kênh NÚT   — "láng giềng nào quan trọng với nút i?"
    e_i^E = kênh CẠNH  — "liên kết nào quanh nút i quan trọng?"
    h_i   = Fusion( h_i^N ‖ e_i^E )

──────────────────────────────────────────────────────────────────────────
SỬA LỖI 1 — Softmax trên kênh cạnh XOÁ MẤT thông tin độ lớn
──────────────────────────────────────────────────────────────────────────
Công thức gốc:   e_i^E = σ( Σ_j β_ij · W_v^E · e_ij ),  Σ_j β_ij = 1

Vì β là softmax nên tổng bằng 1, tức phép này là TRUNG BÌNH CÓ TRỌNG SỐ.
Hệ quả: một nút có 3 liên kết xấu và một nút có 30 liên kết xấu cho ra
CÙNG một véc-tơ e_i^E. Toàn bộ thông tin về SỐ LƯỢNG bị chuẩn hoá mất.

Đây đúng là lỗi đã gặp trên bộ RPL-IDS-Beh, nơi phép lấy trung bình láng
giềng xoá mất chữ ký của tấn công chuyển tiếp chọn lọc — vốn là quan hệ
"bỏ gói CỦA AI" chứ không phải "trung bình bao nhiêu".

Cách sửa: nối thêm một thành phần có mang độ lớn.
    e_i^E = [ Σ_j β_ij W_v e_ij  ‖  log(1 + deg_i) · mean_j(W_v e_ij) ]
Cờ: edge_agg = "mean" (như gốc) | "mean_deg" (mặc định, đã sửa)

──────────────────────────────────────────────────────────────────────────
SỬA LỖI 2 — Điểm chú ý của kênh cạnh LẶP LẠI đúng phép trộn mà tài liệu
             phê phán
──────────────────────────────────────────────────────────────────────────
Công thức gốc:   z_ij^E = (a^E)ᵀ LeakyReLU( W^E [ h_i ‖ h_j ‖ e_ij ] )

Đây CHÍNH LÀ công thức của mô hình nền hiện tại. Nếu giả thuyết HP2 là
"đặc trưng cạnh bị lấn át khi trộn chung với đặc trưng nút", thì kênh cạnh
đang tái tạo lại đúng vấn đề đó ở ngay bước tính điểm.

Cách sửa: cho phép tính điểm chú ý CHỈ từ trạng thái cạnh, để thật sự cô
lập kênh cạnh khỏi ảnh hưởng của biểu diễn nút.
    edge_score = "full"      -> z^E = a^E·LReLU(W^E[h_i ‖ h_j ‖ e_ij])   (như gốc)
    edge_score = "edge_only" -> z^E = a^E·LReLU(W^E e_ij)                (đã sửa)
Chạy CẢ HAI để biết HP2 đúng hay sai — đây mới là phép thử của giả thuyết.

──────────────────────────────────────────────────────────────────────────
SỬA LỖI 3 — Tự vòng không có đặc trưng cạnh
──────────────────────────────────────────────────────────────────────────
Tầng GATv2 của mô hình nền dùng add_self_loops=True, nhưng cạnh tự vòng
(i,i) KHÔNG có véc-tơ đặc trưng tương ứng trong dữ liệu. Tài liệu không đề
cập trường hợp này. Nếu để mặc định, thư viện sẽ đệm bằng 0 và nút cô lập
nhận e_i^E = 0, không phân biệt được với nút có mọi liên kết đều "sạch".

Cách sửa: học một véc-tơ tự-cạnh e_self ∈ R^{d_e}, dùng cho mọi tự vòng.

──────────────────────────────────────────────────────────────────────────
SỬA LỖI 4 — Số tham số: tài liệu khẳng định TĂNG, thực tế GIẢM
──────────────────────────────────────────────────────────────────────────
Tài liệu viết "số tham số chắc chắn tăng so với baseline". Tính lại theo
đúng cách PyG cài GATv2Conv (lin_l, lin_r, lin_edge, att):

    Baseline EdgeGAT (RADAR)  ≈ 103.000   (thực đo 103.956)
    NE-GAT-IDS đề xuất        ≈  61.000   -> ÍT hơn khoảng 41%

Nguyên nhân: mô hình nền dùng 3 đầu chú ý nối lại nên luồng ẩn ở hai tầng
giữa rộng 144 chiều, trong khi NE-GAT giữ 48 chiều xuyên suốt.

Hệ quả cho việc diễn giải kết quả — QUAN TRỌNG:
  · Nếu NE-GAT THUA baseline: chưa chắc do kiến trúc, có thể chỉ do dung
    lượng nhỏ hơn 41%. Phải chạy thêm một biến thể NE-GAT nâng chiều cho
    tương đương tham số rồi mới kết luận.
  · Nếu NE-GAT THẮNG baseline: kết quả MẠNH HƠN nhiều so với tài liệu dự
    kiến, vì nó thắng với ít tham số hơn hẳn.
Lớp này in ra số tham số của từng khối để kiểm tra ngay khi khởi tạo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import degree

from .edge_gat import DyT   # dùng lại Dynamic Tanh của mô hình nền


class NodeChannel(nn.Module):
    """Kênh NÚT: GATv2 KHÔNG dùng đặc trưng cạnh (đúng theo HP1)."""

    def __init__(self, d_in, d_out, heads, dropout):
        super().__init__()
        assert d_out % heads == 0, \
            f"d_out ({d_out}) phải chia hết cho heads ({heads}) để nối lại đúng chiều"
        self.conv = GATv2Conv(d_in, d_out // heads, heads=heads,
                              dropout=dropout, edge_dim=None,
                              concat=True, add_self_loops=True)

    def forward(self, h, edge_index):
        return self.conv(h, edge_index)


class EdgeChannel(nn.Module):
    """Kênh CẠNH: tổng hợp đặc trưng LIÊN KẾT quanh mỗi nút.

    Trả về e_i^E — "ngữ cảnh liên kết" của nút i.
    """

    def __init__(self, d_node, d_edge, dropout,
                 score_mode="full", agg="mean_deg", neg_slope=0.2):
        super().__init__()
        self.score_mode = score_mode
        self.agg = agg
        self.d_edge = d_edge

        # ── điểm chú ý ──
        if score_mode == "edge_only":
            # SỬA LỖI 2: chỉ dùng trạng thái cạnh, cô lập khỏi biểu diễn nút
            self.lin_score = nn.Linear(d_edge, d_edge)
        else:
            self.lin_score = nn.Linear(2 * d_node + d_edge, d_edge)
        self.att = nn.Parameter(torch.empty(1, d_edge))
        nn.init.xavier_uniform_(self.att)

        self.lin_val = nn.Linear(d_edge, d_edge)
        # SỬA LỖI 3: véc-tơ đặc trưng học được cho cạnh tự vòng
        self.e_self = nn.Parameter(torch.zeros(1, d_edge))
        nn.init.normal_(self.e_self, std=0.02)
        self.neg_slope = neg_slope
        self.drop = nn.Dropout(dropout)

    def out_dim(self):
        return self.d_edge * (2 if self.agg == "mean_deg" else 1)

    def forward(self, h, edge_index, e):
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # SỬA LỖI 3: bổ sung tự vòng kèm véc-tơ cạnh học được
        loop = torch.arange(N, device=h.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        e = torch.cat([e, self.e_self.expand(N, -1)], dim=0)

        if self.score_mode == "edge_only":
            z = self.lin_score(e)
        else:
            z = self.lin_score(torch.cat([h[dst], h[src], e], dim=-1))
        z = F.leaky_relu(z, self.neg_slope)
        logit = (z * self.att).sum(-1)                       # [M]

        # softmax theo từng nút đích
        m = logit.max()
        ex = torch.exp(logit - m)
        denom = torch.zeros(N, device=h.device).index_add_(0, dst, ex) + 1e-16
        beta = self.drop(ex / denom[dst])                    # [M]

        val = self.lin_val(e)                                # [M, d_edge]
        out = torch.zeros(N, self.d_edge, device=h.device)
        out.index_add_(0, dst, beta.unsqueeze(-1) * val)     # trung bình có trọng số

        if self.agg == "mean_deg":
            # SỬA LỖI 1: bổ sung thành phần MANG ĐỘ LỚN.
            # Trung bình có trọng số ở trên bất biến với số lượng liên kết;
            # nhân thêm log(1+bậc) khôi phục thông tin "bao nhiêu liên kết".
            deg = degree(dst, num_nodes=N).clamp(min=1).unsqueeze(-1)
            mean_val = torch.zeros(N, self.d_edge, device=h.device)
            mean_val.index_add_(0, dst, val)
            mean_val = mean_val / deg
            out = torch.cat([out, torch.log1p(deg) * mean_val], dim=-1)
        return out


class NEGATLayer(nn.Module):
    """Một tầng NE-GAT: hai kênh song song -> hợp nhất -> phần dư."""

    def __init__(self, d_node, d_edge, node_heads, dropout,
                 score_mode, agg, edge_update, use_dyt=True):
        super().__init__()
        self.node = NodeChannel(d_node, d_node, node_heads, dropout)
        self.edge = EdgeChannel(d_node, d_edge, dropout, score_mode, agg)
        self.fuse = nn.Linear(d_node + self.edge.out_dim(), d_node)
        self.norm = DyT(d_node) if use_dyt else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.edge_update = edge_update
        if edge_update:
            self.eu = nn.Linear(d_edge + 2 * d_node, d_edge)
            self.eu_norm = DyT(d_edge) if use_dyt else nn.Identity()

    def forward(self, h, edge_index, e):
        hN = self.node(h, edge_index)                 # [N, d_node]
        eE = self.edge(h, edge_index, e)              # [N, out_dim]
        fused = self.fuse(torch.cat([hN, eE], dim=-1))
        h_new = self.drop(self.norm(fused + h))       # phần dư từng tầng

        if self.edge_update:
            src, dst = edge_index[0], edge_index[1]
            e = self.eu_norm(self.eu(torch.cat([e, h[dst], h[src]], dim=-1)))
        return h_new, e


class NEGAT(nn.Module):
    """NE-GAT-IDS đầy đủ."""

    def __init__(self, in_dim, edge_dim, num_classes,
                 hidden_dim=48, edge_latent=24, out_dim=32,
                 node_heads=2, num_layers=3, dropout=0.3,
                 classifier_type="mlp", clf_hidden=128,
                 edge_score_mode="full", edge_agg="mean_deg",
                 edge_update=True, use_dyt=True, verbose=True):
        super().__init__()
        self.cfg = dict(edge_score_mode=edge_score_mode, edge_agg=edge_agg,
                        edge_update=edge_update, node_heads=node_heads,
                        num_layers=num_layers)

        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_latent)
        self.node_norm = DyT(hidden_dim) if use_dyt else nn.Identity()
        self.edge_norm = DyT(edge_latent) if use_dyt else nn.Identity()

        self.layers = nn.ModuleList([
            NEGATLayer(hidden_dim, edge_latent, node_heads, dropout,
                       edge_score_mode, edge_agg,
                       edge_update and (l < num_layers - 1),  # tầng cuối không cần
                       use_dyt)
            for l in range(num_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.out_norm = DyT(out_dim) if use_dyt else nn.Identity()

        if classifier_type == "mlp":
            self.clf = nn.Sequential(
                nn.Linear(out_dim, clf_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(clf_hidden, clf_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(clf_hidden // 2, num_classes))
        else:
            self.clf = nn.Linear(out_dim, num_classes)

        if verbose:
            self._report()

    def _report(self):
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        parts = {
            "chiếu đầu vào": cnt(self.node_proj) + cnt(self.edge_proj),
            "kênh NÚT": sum(cnt(l.node) for l in self.layers),
            "kênh CẠNH": sum(cnt(l.edge) for l in self.layers),
            "hợp nhất": sum(cnt(l.fuse) for l in self.layers),
            "cập nhật cạnh": sum(cnt(l.eu) for l in self.layers if l.edge_update),
            "bộ phân loại": cnt(self.out_proj) + cnt(self.clf),
        }
        tot = sum(p.numel() for p in self.parameters())
        print(f"  NE-GAT-IDS: {tot:,} tham số  |  " +
              " · ".join(f"{k} {v:,}" for k, v in parts.items() if v))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False,
                return_feat=False, return_embeddings=False):
        """return_feat=True: trả (logits, z) với z là biểu diễn ẩn TRƯỚC bộ
        phân loại, CÒN GIỮ đồ thị tính toán — cần cho hàm mất mát ATA.
        return_embeddings=True: trả dict đã detach, dùng cho t-SNE."""
        h = F.elu(self.node_norm(self.node_proj(x)))
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1),
                                    self.edge_proj.in_features, device=x.device)
        e = F.elu(self.edge_norm(self.edge_proj(edge_attr)))
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        h = F.elu(self.out_norm(self.out_proj(h)))
        logits = self.clf(h)
        if return_feat:
            return logits, h
        if return_embeddings:
            return {"final": h.detach(), "logits": logits.detach()}
        return logits


def build_negat(cfg, meta):
    m = cfg["model"]
    return NEGAT(
        in_dim=meta["n_node_features"], edge_dim=meta["n_edge_features"],
        num_classes=meta["num_classes"],
        hidden_dim=m.get("hidden_dim", 48),
        edge_latent=m.get("edge_latent", 24),
        out_dim=m.get("out_dim", 32),
        node_heads=m.get("node_heads", 2),
        num_layers=m.get("num_layers", 3),
        dropout=m.get("dropout", 0.3),
        classifier_type=m.get("classifier_type", "mlp"),
        clf_hidden=m.get("clf_hidden", 128),
        edge_score_mode=m.get("edge_score_mode", "full"),
        edge_agg=m.get("edge_agg", "mean_deg"),
        edge_update=m.get("edge_update", True),
        use_dyt=m.get("use_dyt", True),
    )
