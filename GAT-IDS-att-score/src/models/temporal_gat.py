# -*- coding: utf-8 -*-
"""
temporal_gat.py — T-GAT-IDS: mở rộng ĐỘNG THEO THỜI GIAN
=========================================================

VẤN ĐỀ ĐANG GIẢI QUYẾT
----------------------
Ba bộ dữ liệu của đề tài đều là ĐỒ THỊ ĐỘNG RỜI RẠC: mỗi cửa sổ sinh một
ảnh chụp riêng, và giữa các ảnh chụp thì cả đặc trưng nút lẫn tập cạnh đều
thay đổi. Nhưng cả GAT-IDS lẫn NE-GAT-IDS đều xử lý mỗi ảnh chụp ĐỘC LẬP:
không có cạnh nào nối nút i ở cửa sổ t với chính nó ở cửa sổ t+1.

Thông tin xuyên cửa sổ duy nhất hiện có là nhóm đặc trưng D, dùng trung
bình trượt luỹ thừa — một phép nén bốn chiều của toàn bộ lịch sử. Hệ quả
đo được: lớp delayed_reply có F1 thấp nhất ở CẢ HAI mô hình (0,625 và
0,648), và chữ ký của nó vốn phụ thuộc TRÌNH TỰ sự kiện.

Ý TƯỞNG
-------
Giữ nguyên bộ mã hoá không gian, bổ sung một tầng CHÚ Ý THỜI GIAN chạy
trên chuỗi biểu diễn của CÙNG MỘT NÚT qua các cửa sổ liên tiếp:

    ảnh chụp t-W+1 ──┐
    ...              ├─ NE-GAT (không gian) ─→ h_i^(t-W+1..t)
    ảnh chụp t     ──┘                              │
                                                    ▼
                                    chú ý thời gian NHÂN QUẢ
                                                    │
                                                    ▼
                                          phân loại nút ở cửa sổ t

BA RÀNG BUỘC THIẾT KẾ
---------------------
1. NHÂN QUẢ TUYỆT ĐỐI. Biểu diễn tại cửa sổ t chỉ được nhìn các cửa sổ
   <= t. Đây không phải lựa chọn tinh chỉnh mà là điều kiện đúng đắn: cho
   phép nhìn tương lai chính là tái tạo lại đúng dạng rò rỉ thời gian mà
   toàn bộ giao thức đánh giá của đề tài nhằm loại bỏ. Mặt nạ tam giác
   trên được áp cứng trong mã, không có cờ để tắt.

2. CHUỖI DỰNG TRONG TỪNG TỆP. Mỗi tệp là một lần chạy mô phỏng độc lập
   với đồng hồ riêng bắt đầu từ 0, nên không nối cửa sổ của hai tệp khác
   nhau. Ranh giới tệp là ranh giới chuỗi.

3. NÚT VẮNG MẶT ĐƯỢC ĐỆM VÀ CHE. Một thiết bị có thể không phát gói nào
   trong một cửa sổ nên không xuất hiện trong ảnh chụp đó. Vị trí tương
   ứng được đệm bằng véc-tơ không và che bằng key_padding_mask, thay vì
   nội suy — nội suy sẽ bịa ra hành vi không quan sát được.

SO VỚI CÁC HƯỚNG KHÁC
---------------------
Hướng thay thế là dựng một đồ thị thời gian-không gian duy nhất với cạnh
thời gian nối (i,t) với (i,t+1), rồi chạy GNN trên đó. Hướng đó biểu đạt
mạnh hơn nhưng làm đồ thị phình theo W lần và trộn lẫn hai loại quan hệ
có ngữ nghĩa khác hẳn nhau vào cùng một cơ chế chú ý. Cách tách hai tầng
ở đây giữ được khả năng đo riêng đóng góp của từng chiều.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ne_gat import NEGAT
from .edge_gat import EdgeAwareGAT


# ══════════════════════════════════════════════════════════════════
class CausalTemporalAttention(nn.Module):
    """Chú ý thời gian một chiều trên chuỗi biểu diễn của một nút."""

    def __init__(self, d_in, d_model=64, nhead=4, num_layers=1,
                 dim_ff=128, dropout=0.1, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        # Mã vị trí học được: thứ tự cửa sổ có nghĩa, và số cửa sổ hữu hạn
        # nên không cần mã hình sin.
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_in)
        self.norm = nn.LayerNorm(d_in)
        self.max_len = max_len

    @staticmethod
    def _causal(T, device):
        """Mặt nạ tam giác trên: vị trí t KHÔNG được nhìn t' > t."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                          diagonal=1)

    def forward(self, seq, pad_mask=None):
        """seq: [B, T, d_in] — B chuỗi nút, T cửa sổ. Trả [B, T, d_in]."""
        B, T, _ = seq.shape
        z = self.in_proj(seq) + self.pos[:, :T]
        z = self.enc(z, mask=self._causal(T, seq.device),
                     src_key_padding_mask=pad_mask)
        # Phần dư: giữ đường đi cho biểu diễn KHÔNG GIAN thuần tuý, để tầng
        # thời gian chỉ cần học phần BỔ SUNG thay vì tái tạo lại từ đầu.
        return self.norm(seq + self.out_proj(z))


# ══════════════════════════════════════════════════════════════════
def build_node_sequences(graphs, embeds, window=8):
    """Dựng chuỗi thời gian cho từng nút, trong từng tệp mô phỏng.

    graphs : danh sách Data đã sắp theo (src_file, win_idx)
    embeds : danh sách tensor [N_g, d] — embedding không gian của từng đồ thị

    Trả về:
      seq      [M, window, d]  chuỗi của M vị trí (nút, cửa sổ) đích
      pad      [M, window]     True ở vị trí KHÔNG có quan sát
      target   [M]             chỉ số phẳng của vị trí đích trong embeds
    """
    by_file = {}
    for gi, g in enumerate(graphs):
        by_file.setdefault(getattr(g, "src_file", "_"), []).append(gi)

    d = embeds[0].size(1)
    offset, offs = 0, []
    for e in embeds:
        offs.append(offset); offset += e.size(0)
    flat = torch.cat(embeds, 0)

    seqs, pads, tgts = [], [], []
    for _f, gis in by_file.items():
        gis = sorted(gis, key=lambda i: int(getattr(graphs[i], "win_idx", i)))
        # vị trí của từng node_id trong từng cửa sổ của tệp này
        pos = []
        for gi in gis:
            ids = getattr(graphs[gi], "node_ids", None)
            if ids is None:
                ids = torch.arange(graphs[gi].num_nodes)
            pos.append({int(v): k for k, v in enumerate(ids.tolist())})

        for w, gi in enumerate(gis):
            ids = list(pos[w].keys())
            lo = max(0, w - window + 1)
            hist = list(range(lo, w + 1))
            for nid in ids:
                s = flat.new_zeros(window, d)
                p = torch.ones(window, dtype=torch.bool)
                for k, hw in enumerate(hist):
                    slot = window - len(hist) + k      # căn PHẢI: đích ở cuối
                    j = pos[hw].get(nid)
                    if j is not None:
                        s[slot] = flat[offs[gis[hw]] + j]
                        p[slot] = False
                seqs.append(s); pads.append(p)
                tgts.append(offs[gi] + pos[w][nid])
    return (torch.stack(seqs), torch.stack(pads),
            torch.tensor(tgts, dtype=torch.long))


# ══════════════════════════════════════════════════════════════════
class TemporalGATIDS(nn.Module):
    """T-GAT-IDS = bộ mã hoá KHÔNG GIAN + chú ý THỜI GIAN nhân quả."""

    def __init__(self, in_dim, edge_dim, num_classes, backbone="negat",
                 hidden_dim=48, edge_latent=24, out_dim=32, node_heads=2,
                 heads=3, num_layers=3, dropout=0.3, clf_hidden=128,
                 t_model=64, t_head=4, t_layers=1, t_ff=128, t_dropout=0.1,
                 window=8, edge_score_mode="full", edge_agg="mean_deg",
                 edge_update=True, use_edge_features=True, verbose=True):
        super().__init__()
        self.window = window
        self.backbone_name = backbone

        if backbone == "negat":
            self.gnn = NEGAT(
                in_dim=in_dim, edge_dim=edge_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, edge_latent=edge_latent, out_dim=out_dim,
                node_heads=node_heads, num_layers=num_layers, dropout=dropout,
                classifier_type="linear", encoder_only=True,
                edge_score_mode=edge_score_mode, edge_agg=edge_agg,
                edge_update=edge_update, verbose=False)
        else:
            self.gnn = EdgeAwareGAT(
                in_dim=in_dim, edge_dim=edge_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, out_dim=out_dim, heads=heads,
                num_layers=num_layers, dropout=dropout,
                use_edge_features=use_edge_features, classifier_type="linear",
                layer_residual=True, skip_mode="none", use_main_residual=False)

        self.temporal = CausalTemporalAttention(
            d_in=out_dim, d_model=t_model, nhead=t_head,
            num_layers=t_layers, dim_ff=t_ff, dropout=t_dropout,
            max_len=max(window, 8))

        self.clf = nn.Sequential(
            nn.Linear(out_dim, clf_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden, clf_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden // 2, num_classes))
        if verbose:
            self._report()

    def _report(self):
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        print(f"  T-GAT-IDS ({self.backbone_name}): "
              f"{sum(p.numel() for p in self.parameters()):,} tham số  |  "
              f"không gian {cnt(self.gnn):,} · thời gian {cnt(self.temporal):,} "
              f"· phân loại {cnt(self.clf):,}  |  cửa sổ W={self.window}")

    def embed_graph(self, x, edge_index, edge_attr):
        out = self.gnn(x, edge_index, edge_attr, return_feat=True)
        return out[1] if isinstance(out, tuple) else out

    def forward_sequences(self, seq, pad, target_idx=None):
        """seq [M,W,d], pad [M,W] -> logits tại vị trí CUỐI (thời điểm đích)."""
        ctx = self.temporal(seq, pad_mask=pad)
        return self.clf(ctx[:, -1, :])

    def forward(self, x, edge_index, edge_attr=None, **kw):
        """Chế độ một ảnh chụp: chuỗi độ dài 1, tầng thời gian thoái hoá về
        phép biến đổi đồng nhất cộng phần dư. Dùng để kiểm thử nhanh."""
        h = self.embed_graph(x, edge_index, edge_attr)
        seq = h.unsqueeze(1)
        pad = torch.zeros(h.size(0), 1, dtype=torch.bool, device=h.device)
        return self.forward_sequences(seq, pad)


def build_temporal_gat(cfg, meta):
    m = cfg["model"]; t = cfg.get("temporal", {})
    return TemporalGATIDS(
        in_dim=meta["n_node_features"], edge_dim=meta["n_edge_features"],
        num_classes=meta["num_classes"],
        backbone=t.get("backbone", "negat"),
        hidden_dim=m.get("hidden_dim", 48), edge_latent=m.get("edge_latent", 24),
        out_dim=m.get("out_dim", 32), node_heads=m.get("node_heads", 2),
        heads=m.get("heads", 3), num_layers=m.get("num_layers", 3),
        dropout=m.get("dropout", 0.3), clf_hidden=m.get("clf_hidden", 128),
        t_model=t.get("d_model", 64), t_head=t.get("nhead", 4),
        t_layers=t.get("layers", 1), t_ff=t.get("dim_ff", 128),
        t_dropout=t.get("dropout", 0.1), window=t.get("window", 8),
        edge_score_mode=m.get("edge_score_mode", "full"),
        edge_agg=m.get("edge_agg", "mean_deg"),
        edge_update=m.get("edge_update", True),
        use_edge_features=m.get("use_edge_features", True))
