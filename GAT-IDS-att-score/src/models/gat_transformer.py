# -*- coding: utf-8 -*-
"""
gat_transformer.py — GAT-Transformer IDS
=========================================

Mô hình lai giữa NE-GAT-IDS và bộ tự mã hoá có che Transformer, phỏng theo
kiến trúc GraphIDS của Guerra và cộng sự (NeurIPS 2025), với một thay thế
then chốt: bộ mã hoá đồ thị E-GraphSAGE được thay bằng NE-GAT-IDS.

    GraphIDS gốc :  E-GraphSAGE  ->  Transformer che  ->  lỗi tái tạo
    Mô hình này  :  NE-GAT-IDS   ->  Transformer che  ->  lỗi tái tạo
                                                       -> (tuỳ chọn) đầu phân loại

BA KHÁC BIỆT SO VỚI BÀI BÁO GỐC, đều bắt nguồn từ bài toán chứ không phải
từ sở thích thiết kế. Cả ba cần nêu rõ khi báo cáo.

1. ĐƠN VỊ PHÂN LOẠI. GraphIDS đặt nút là máy chủ và cạnh là luồng, rồi phân
   loại CẠNH. Bài toán RPL đặt nút là thiết bị và phân loại NÚT, vì yêu cầu
   vận hành là chỉ đích danh thiết bị nào đang tấn công. Do đó Transformer ở
   đây làm việc trên lô biểu diễn NÚT thay vì lô biểu diễn luồng.

2. BÀI TOÁN ĐÍCH. GraphIDS là bộ PHÁT HIỆN BẤT THƯỜNG: điểm số là một vô
   hướng, nên nó chỉ trả lời có tấn công hay không. Đề tài cần phân loại
   16 lớp. Một lỗi tái tạo vô hướng KHÔNG THỂ sinh ra 16 lớp. Vì vậy mô
   hình này dùng công thức hai giai đoạn chuẩn mực của học tự giám sát:

       Giai đoạn 1 — TIỀN HUẤN LUYỆN tự giám sát, CHỈ trên nút bình thường,
                     mục tiêu tái tạo, KHÔNG dùng nhãn tấn công.
       Giai đoạn 2 — TINH CHỈNH có giám sát với đầu phân loại trên dữ liệu
                     có nhãn, khởi tạo từ trọng số giai đoạn 1.

   Giai đoạn 1 một mình vẫn dùng được như bộ phát hiện bất thường nhị phân,
   và đó là chế độ so sánh trực tiếp được với bài báo gốc.

3. KHÔNG CÓ LẤY MẪU LÁNG GIỀNG. GraphIDS dùng lấy mẫu láng giềng vì đồ thị
   của nó rất lớn. Đồ thị DODAG một cửa sổ chỉ có vài chục nút nên không cần,
   và việc bỏ lấy mẫu làm biểu diễn tất định hơn.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ne_gat import NEGAT
from .edge_gat import EdgeAwareGAT


# ══════════════════════════════════════════════════════════════════
#  Bộ tự mã hoá có che
# ══════════════════════════════════════════════════════════════════
class MaskedAutoencoder(nn.Module):
    """Transformer mã hoá–giải mã tái tạo biểu diễn nút.

    Điểm khác biệt so với bộ tự mã hoá có che thông thường: mặt nạ được áp
    TRỰC TIẾP lên ma trận chú ý chứ không xoá token đầu vào. Nó hoạt động như
    một dạng dropout có cấu trúc trên trọng số chú ý, buộc mô hình suy ra từ
    ngữ cảnh một phần thay vì phụ thuộc vào một vài liên kết ngữ cảnh cụ thể.
    Mặt nạ được TẮT khi suy luận.

    KHÔNG dùng mã hoá vị trí: các nút trong một cửa sổ không có thứ tự tự
    nhiên, nên thêm vị trí sẽ đưa vào một tín hiệu giả. Bài báo gốc cũng
    không dùng, và tự chú ý vốn bất biến hoán vị nên điều này là nhất quán.
    """

    def __init__(self, d_gnn, d_model=32, nhead=4, num_layers=2,
                 dim_ff=64, dropout=0.1, mask_ratio=0.3):
        """d_model PHẢI <= d_gnn.

        Bài báo gốc chiếu embedding của GNN xuống một không gian THẤP CHIỀU
        HƠN. Đây không phải chi tiết tiết kiệm tính toán mà là điều kiện để
        phương pháp hoạt động: nếu d_model > d_gnn thì bộ tự mã hoá không
        có NÚT THẮT nào, nên nó học được hàm ĐỒNG NHẤT — tái tạo hoàn hảo
        MỌI đầu vào, kể cả nút tấn công. Khi đó sai số tái tạo tiến về 0 với
        tất cả mẫu và điểm bất thường mất sạch khả năng phân biệt.

        Dấu hiệu nhận biết trong nhật ký: MSE tụt về 0,00000 và ROC-AUC
        rơi xuống quanh hoặc dưới 0,5.
        """
        super().__init__()
        self.d_gnn, self.d_model = d_gnn, d_model
        self.mask_ratio = mask_ratio

        if d_model > d_gnn:
            import warnings
            warnings.warn(
                f"d_model={d_model} > d_gnn={d_gnn}: bộ tự mã hoá KHÔNG có nút "
                f"thắt nên có thể sụp về hàm đồng nhất, khiến điểm bất thường "
                f"mất khả năng phân biệt. Đặt d_model <= {d_gnn}.", RuntimeWarning)
        self.in_proj = nn.Linear(d_gnn, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_gnn)

    def _attn_mask(self, n, device):
        """Mặt nạ nhị phân ĐỐI XỨNG, vô hiệu hoá một tập ngẫu nhiên các liên
        kết chú ý. Đường chéo luôn được giữ để mỗi vị trí còn tự chú ý."""
        m = torch.rand(n, n, device=device) < self.mask_ratio
        m = m | m.T                      # đối xứng
        m.fill_diagonal_(False)          # luôn giữ tự chú ý
        return m

    def forward(self, h, window=512, batch_vec=None):
        """h: [N, d_gnn] -> tái tạo [N, d_gnn].

        XÁO TRỘN Ở ĐÂY LÀ GÌ — và KHÔNG phải là gì
        ------------------------------------------
        Phép hoán vị chỉ đổi THỨ TỰ các véc-tơ biểu diễn trước khi cắt cửa
        sổ. Nó KHÔNG trộn đặc trưng giữa các nút: véc-tơ h_i vẫn gắn nguyên
        vẹn với nút i, và sau khi tái tạo, hoán vị NGHỊCH ĐẢO khôi phục đúng
        thứ tự gốc để mỗi sai số tái tạo ứng đúng nút của nó.

        Mục đích: quyết định những nút NÀO cùng nằm trong một cửa sổ tự chú
        ý. Không xáo trộn thì mỗi cửa sổ chỉ chứa các nút liền kề nhau trong
        lô, và Transformer sẽ lặp lại đúng việc mà GNN đã làm.

        Phép hoán vị này ĐỘC LẬP HOÀN TOÀN với phép chia tập dữ liệu. Tập
        train/val/test đã được tách theo THỨ TỰ THỜI GIAN từ Bước 1, trước
        khi mô hình nhìn thấy bất kỳ dữ liệu nào. Xáo trộn BÊN TRONG tập
        huấn luyện không tạo ra rò rỉ nào.

        batch_vec: véc-tơ chỉ số đồ thị của PyG. Nếu truyền vào, cửa sổ chỉ
        gom nút TRONG CÙNG một đồ thị (phạm vi "graph"); nếu None, cửa sổ
        gom nút trên toàn lô (phạm vi "batch", mặc định của bài báo gốc).
        """
        if batch_vec is not None:
            return self._forward_per_graph(h, window, batch_vec)
        N = h.size(0)
        dev = h.device
        perm = torch.randperm(N, device=dev) if self.training else torch.arange(N, device=dev)
        hs = h[perm]
        out = torch.empty_like(hs)

        for s in range(0, N, window):
            blk = hs[s:s + window]
            n = blk.size(0)
            z = self.in_proj(blk).unsqueeze(0)          # [1, n, d_model]
            mask = self._attn_mask(n, dev) if (self.training and n > 1) else None
            mem = self.encoder(z, mask=mask)
            rec = self.decoder(z, mem, tgt_mask=mask, memory_mask=mask)
            out[s:s + n] = self.out_proj(rec).squeeze(0)

        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(N, device=dev)
        return out[inv]

    def _forward_per_graph(self, h, window, batch_vec):
        """Phạm vi cửa sổ = TRONG MỘT ĐỒ THỊ.

        Lý do có tuỳ chọn này: ở phạm vi mặc định, một cửa sổ 512 nút gom
        nút từ ~32 đồ thị khác nhau, có thể thuộc các tệp mô phỏng và các
        giai đoạn thời gian khác nhau. Khi đó "đồng xuất hiện toàn cục" mà
        Transformer học được nối những nút CHƯA BAO GIỜ cùng tồn tại trong
        một mạng thật. Phạm vi "graph" loại bỏ nghi vấn đó, đổi lại cửa sổ
        nhỏ hơn nhiều nên ngữ cảnh toàn cục hẹp hơn.
        """
        out = torch.empty_like(h)
        for g in torch.unique(batch_vec):
            idx = (batch_vec == g).nonzero(as_tuple=True)[0]
            out[idx] = self.forward(h[idx], window=window, batch_vec=None)
        return out


# ══════════════════════════════════════════════════════════════════
#  Mô hình đầy đủ
# ══════════════════════════════════════════════════════════════════
class GATTransformerIDS(nn.Module):
    """GAT-Transformer IDS — bộ mã hoá đồ thị + tự mã hoá có che.

    Hai chế độ vận hành:

      · mode="ssl"  — trả (h, h_rec). Điểm bất thường là lỗi tái tạo bình
                      phương từng nút. Huấn luyện CHỈ trên nút bình thường.
      · mode="clf"  — trả logits qua đầu phân loại, dùng cho giai đoạn tinh
                      chỉnh có giám sát.
    """

    def __init__(self, in_dim, edge_dim, num_classes,
                 backbone="negat", hidden_dim=48, out_dim=32,
                 edge_latent=24, node_heads=2, heads=3, num_layers=3,
                 dropout=0.3, edge_score_mode="full", edge_agg="mean_deg",
                 edge_update=True, use_edge_features=True,
                 d_model=32, nhead=4, tr_layers=2, dim_ff=64,
                 tr_dropout=0.1, mask_ratio=0.3, clf_hidden=128,
                 verbose=True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "negat":
            self.gnn = NEGAT(
                in_dim=in_dim, edge_dim=edge_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, edge_latent=edge_latent, out_dim=out_dim,
                node_heads=node_heads, num_layers=num_layers, dropout=dropout,
                classifier_type="linear", encoder_only=True,
                edge_score_mode=edge_score_mode,
                edge_agg=edge_agg, edge_update=edge_update, verbose=False)
        elif backbone == "edgegat":
            self.gnn = EdgeAwareGAT(
                in_dim=in_dim, edge_dim=edge_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, out_dim=out_dim, heads=heads,
                num_layers=num_layers, dropout=dropout,
                use_edge_features=use_edge_features, classifier_type="linear",
                layer_residual=True, skip_mode="none", use_main_residual=False)
        else:
            raise ValueError(f"backbone không hợp lệ: {backbone}")

        # DÒ số chiều thật của biểu diễn mà bộ mã hoá trả về, thay vì giả định
        # nó bằng out_dim. Lý do: EdgeAwareGAT có thể nối thêm đặc trưng thô
        # vào biểu diễn trước bộ phân loại, nên chiều thật lớn hơn out_dim.
        with torch.no_grad():
            _x = torch.zeros(3, in_dim)
            _ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            _ea = torch.zeros(3, edge_dim)
            _, _h = self.gnn(_x, _ei, _ea, return_feat=True)
            d_feat = int(_h.shape[-1])
        self.d_feat = d_feat

        self.mae = MaskedAutoencoder(d_gnn=d_feat, d_model=d_model, nhead=nhead,
                                     num_layers=tr_layers, dim_ff=dim_ff,
                                     dropout=tr_dropout, mask_ratio=mask_ratio)

        # Đầu phân loại cho giai đoạn 2. Nhận NỐI của biểu diễn đồ thị với
        # biểu diễn đã tái tạo: phần dư giữa hai véc-tơ chính là tín hiệu bất
        # thường mà giai đoạn 1 đã học, nên đưa cả hai vào giúp đầu phân loại
        # dùng được tín hiệu đó thay vì phải học lại từ đầu.
        self.clf = nn.Sequential(
            nn.Linear(d_feat * 2, clf_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden, clf_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden // 2, num_classes))

        if verbose:
            self._report()

    def _report(self):
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        tot = sum(p.numel() for p in self.parameters())
        print(f"  GAT-Transformer IDS: {tot:,} tham số  |  chiều biểu diễn "
              f"{self.d_feat}  |  "
              f"bộ mã hoá đồ thị ({self.backbone_name}) {cnt(self.gnn):,} · "
              f"tự mã hoá có che {cnt(self.mae):,} · đầu phân loại {cnt(self.clf):,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── biểu diễn đồ thị mức nút ──
    def embed(self, x, edge_index, edge_attr):
        _, h = self.gnn(x, edge_index, edge_attr, return_feat=True)
        return h

    def forward(self, x, edge_index, edge_attr=None, mode="ssl",
                window=512, return_feat=False, batch_vec=None):
        h = self.embed(x, edge_index, edge_attr)
        h_rec = self.mae(h, window=window, batch_vec=batch_vec)
        if mode == "ssl":
            return h, h_rec
        z = torch.cat([h, h_rec], dim=-1)
        logits = self.clf(z)
        if return_feat:
            return logits, h
        return logits

    @staticmethod
    def anomaly_score(h, h_rec):
        """Điểm bất thường từng nút = lỗi tái tạo bình phương chuẩn L2."""
        return ((h - h_rec) ** 2).sum(dim=-1)


def build_gat_transformer(cfg, meta):
    m = cfg["model"]
    t = cfg.get("transformer", {})
    return GATTransformerIDS(
        in_dim=meta["n_node_features"], edge_dim=meta["n_edge_features"],
        num_classes=meta["num_classes"],
        backbone=m.get("ssl_backbone", "negat"),
        hidden_dim=m.get("hidden_dim", 48), out_dim=m.get("out_dim", 32),
        edge_latent=m.get("edge_latent", 24), node_heads=m.get("node_heads", 2),
        heads=m.get("heads", 3), num_layers=m.get("num_layers", 3),
        dropout=m.get("dropout", 0.3),
        edge_score_mode=m.get("edge_score_mode", "full"),
        edge_agg=m.get("edge_agg", "mean_deg"),
        edge_update=m.get("edge_update", True),
        use_edge_features=m.get("use_edge_features", True),
        d_model=t.get("d_model", 32), nhead=t.get("nhead", 4),
        tr_layers=t.get("layers", 2), dim_ff=t.get("dim_ff", 64),
        tr_dropout=t.get("dropout", 0.1), mask_ratio=t.get("mask_ratio", 0.3),
        clf_hidden=m.get("clf_hidden", 128))
