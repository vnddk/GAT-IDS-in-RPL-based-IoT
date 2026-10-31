"""Mô hình Edge-aware GAT cho phát hiện xâm nhập RPL-IoT.

Đóng góp cốt lõi: dùng GATv2Conv với tham số `edge_dim` để đưa đặc trưng cạnh
(ETX, PDR_link, RSSI, hop, forward_ratio) VÀO hệ số attention. Công thức:

    e_ij = a^T · LeakyReLU( W·[h_i ‖ h_j ‖ edge_ij] )
    α_ij = softmax_j(e_ij)
    h_i' = Σ_j α_ij · W·h_j

So với GAT gốc (không edge): GATv2Conv(edge_dim=None). Cờ use_edge_features
cho phép ablation bật/tắt — đúng KC-2 trong đề cương.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al., CVPR 2025) — drop-in thay cho normalization.

        DyT(x) = gamma * tanh(alpha * x) + beta

    alpha: scalar học được (điều chỉnh scale theo range input).
    gamma, beta: vector per-channel học được (như affine của LayerNorm).
    Ưu điểm cho dữ liệu RPL: tự thích ứng range hẹp, ổn định gradient,
    không phụ thuộc batch statistics (tốt cho federated non-IID batch nhỏ).
    """
    def __init__(self, num_features: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class EdgeAwareGAT(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, out_dim=32,
                 num_classes=6, heads=4, dropout=0.3, use_edge_features=True,
                 use_dyt=True, use_input_skip=True, skip_mode="raw",
                 num_layers=2, layer_residual=False, use_main_residual=True,
                 classifier_type="linear", clf_hidden=64, conv_type: str = "gatv2"):
        """EdgeGAT với số tầng GAT cấu hình được.

        num_layers = L: L tầng GATv2 tương ứng bán kính thu nhận L-hop.
            L−1 tầng đầu dùng concat=True  (hidden_dim × heads)
            tầng cuối       dùng concat=False (out_dim)

        layer_residual: bật phần dư TỪNG TẦNG cho các tầng giữa. Khi tăng số
            tầng, làm mịn quá mức là rủi ro chính; phần dư từng tầng giúp mỗi
            tầng giữ lại biểu diễn trước đó thay vì bị trung bình hoá hết.
            Chỉ có tác dụng khi num_layers ≥ 3.

        use_main_residual: bật/tắt PHẦN DƯ CHÍNH  h_L ← ELU(h_L + W_res·h⁰).

            Đây là đường bắc thẳng từ biểu diễn ĐẦU VÀO tới đầu ra tầng cuối,
            bỏ qua toàn bộ L tầng GAT. Tắt nó KHÔNG làm mất kết nối tắt đặc
            trưng gốc (use_input_skip) — hai cơ chế độc lập:
                phần dư chính : CỘNG h⁰ (đã qua Linear+DyT+ELU) vào h_L
                kết nối tắt   : NỐI X thô vào ngay trước bộ phân loại
            Khi tắt, mô hình vẫn còn ELU cuối để giữ tính phi tuyến nhất quán.

        classifier_type: kiểu tầng phân loại cuối.
            "linear" — Linear(clf_in → C).  Mặc định, ít tham số nhất.
            "mlp"    — MLP HAI tầng ẩn:
                         Linear(clf_in→H) → ReLU → Dropout
                       → Linear(H→H/2)    → ReLU → Dropout
                       → Linear(H/2→C)
                       H đặt bằng clf_hidden (mặc định 128). Cho ranh giới quyết
                       định PHI TUYẾN; hữu ích khi các lớp chồng lấn (nhóm
                       họ-rank của bộ RADAR).
            "lstm"   — LSTM đọc chuỗi biểu diễn CÁC TẦNG  h¹ → h² → ... → h^L,
                       rồi phân loại từ trạng thái ẩn cuối (nối thêm raw-skip
                       nếu bật).

                       LƯU Ý THIẾT KẾ: LSTM ở đây KHÔNG chạy trên các chiều của
                       vector đặc trưng — thứ tự các chiều là tuỳ ý nên coi
                       chúng là chuỗi là sai nguyên lý. Thay vào đó LSTM chạy
                       trên chuỗi TẦNG, vốn có thứ tự vật lý thật: mỗi bước ứng
                       với bán kính thu nhận 1-hop, 2-hop, 3-hop. Đây là biến
                       thể LSTM của Jumping Knowledge Network (Xu và cộng sự,
                       ICML 2018), cho phép mỗi node tự chọn bán kính phù hợp.
                       Chỉ có ý nghĩa khi num_layers ≥ 2.
        """
        super().__init__()
        assert num_layers >= 1, "num_layers phải ≥ 1"
        self.use_edge = use_edge_features
        self.use_dyt = use_dyt
        self.use_input_skip = use_input_skip
        self.num_layers = int(num_layers)
        self.layer_residual = bool(layer_residual)
        self.use_main_residual = bool(use_main_residual)
        self.classifier_type = str(classifier_type).lower()
        ed = edge_dim if use_edge_features else None

        # ── conv_type: "gatv2" (mặc định) hoặc "sage" ─────────────────────
        # "sage" thay tầng chú ý bằng SAGEConv tổng hợp TRUNG BÌNH — mọi láng
        # giềng có trọng số như nhau. Dùng cho ablation 2×2 ở
        # scripts/ablation_edge_attention.py: nó cho phép TẮT cơ chế chú ý mà
        # giữ nguyên toàn bộ phần còn lại (DyT, phần dư, bộ phân loại, hàm mất
        # mát), nên chênh lệch đo được quy đúng về chú ý chứ không lẫn với
        # khác biệt kiến trúc như khi so với GCN hay E-GraphSAGE.
        self.conv_type = str(conv_type).lower()
        if self.conv_type == "sage" and ed is not None:
            # SAGEConv không nhận edge_attr; đặc trưng cạnh khi đó phải được
            # gộp vào đặc trưng NÚT trước khi đưa vào model (xem script ablation).
            ed = None

        def _norm(d):
            return DyT(d) if use_dyt else nn.BatchNorm1d(d)

        # Input projection + DyT (thay BatchNorm)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.dyt_in = _norm(hidden_dim)

        # ── Chồng num_layers tầng GATv2 ──
        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()
        wide = hidden_dim * heads
        d_cur = hidden_dim
        for li in range(self.num_layers):
            last = (li == self.num_layers - 1)
            d_out = out_dim if last else hidden_dim
            if self.conv_type == "sage":
                from torch_geometric.nn import SAGEConv
                # để giữ ĐÚNG số chiều như nhánh GATv2 (concat 3 đầu ở tầng
                # giữa), tầng sage xuất d_out*heads ở tầng không phải cuối
                self.gats.append(SAGEConv(d_cur, d_out if last else wide,
                                          aggr="mean"))
            else:
                self.gats.append(GATv2Conv(
                    d_cur, d_out, heads=heads, dropout=dropout,
                    edge_dim=ed, concat=(not last), add_self_loops=True,
                ))
            d_cur = d_out if last else wide
            self.norms.append(_norm(d_cur))

        # Phần dư TỪNG TẦNG cho các tầng giữa (tuỳ chọn, chống làm mịn quá mức)
        self.mid_res = None
        if self.layer_residual and self.num_layers >= 3:
            # tầng 2..L−1 nhận phần dư từ đầu ra tầng trước (cùng là `wide`)
            self.mid_res = nn.ModuleList([
                nn.Identity() for _ in range(self.num_layers - 2)
            ])

        # Residual chính: h0 (hidden_dim) -> đầu ra tầng cuối (out_dim)
        # Chỉ khởi tạo khi BẬT, để checkpoint không chứa tham số không dùng.
        self.res_proj = (nn.Linear(hidden_dim, out_dim)
                         if self.use_main_residual else None)

        # ── SKIP CONNECTION ────────────────────────────────────────────────
        # v3.10: BỎ nhánh "hybrid" (Jumping-Knowledge từ tầng 1).
        # Lý do: silhouette đo trên tập test cho h2 = 0,2555 nhưng final = 0,2216
        # — nối thêm nhánh JK LÀM GIẢM khả năng tách cụm 0,034. Nhánh này đưa vào
        # nhiễu nhiều hơn thông tin. Giữ lại raw-skip (đã chứng minh hiệu quả:
        # worst_parent 0,154 -> 0,878 khi bật).
        #
        # Chế độ "hybrid" vẫn còn trong code để tái lập thí nghiệm ablation,
        # nhưng KHÔNG còn là mặc định.
        # (Jumping Knowledge gốc: Xu và cs., ICML 2018)
        # skip_mode:
        #   "none"   : không skip nào (chỉ residual h0 -> h2)
        #   "raw"    : raw-skip — nối đặc trưng GỐC x (mặc định cũ)
        #   "hybrid" : raw-skip + JK từ TẦNG 1 (giữ biểu diễn TRUNG GIAN)
        #
        # VÌ SAO: residual hiện tại chỉ nối h0 (trước mọi lan truyền) với h2
        # (sau HAI vòng). Biểu diễn TẦNG 1 — đã trộn láng giềng 1-hop nhưng
        # CHƯA bị làm mịn hai lần — không có đường nào tới bộ phân loại.
        # JKNet gọi đây là "layer aggregation": gộp biểu diễn của MỌI tầng
        # để giữ tính cục bộ, chống over-smoothing.
        if skip_mode == "hybrid":
            import warnings
            warnings.warn("skip_mode='hybrid' đã bị loại khỏi khuyến nghị v3.10 "
                          "(silhouette final < h2). Chỉ dùng để tái lập ablation.",
                          stacklevel=2)
        self.skip_mode = skip_mode
        self.jk_proj = None
        if skip_mode == "hybrid":
            self.jk_proj = nn.Linear(hidden_dim * heads, out_dim)
            # cổng học được cho nhánh JK: sigmoid(0)=0.5 -> khởi động trung tính
            self.jk_gate = nn.Parameter(torch.zeros(1))

        self.dropout = dropout
        # Raw-feature skip: classifier nhìn CẢ embedding GNN VÀ features gốc.
        # Lý do: với lớp siêu hiếm (time_based, ~0.17% node), message passing
        # làm loãng chữ ký per-node của attacker vào láng giềng. Concat features
        # gốc cho classifier khả năng "tree-like" khai thác trực tiếp thống kê
        # node (vd. tỉ lệ forward giảm ở worst_parent) song song với topology.
        clf_in = out_dim + (in_dim if use_input_skip else 0)
        if skip_mode == "hybrid":
            clf_in += out_dim                     # thêm nhánh JK từ tầng 1
        # ── Tầng phân loại: linear | mlp | lstm ──
        ct = self.classifier_type
        if ct == "mlp":
            # MLP HAI tầng ẩn: clf_in -> H -> H/2 -> C
            # Tầng thứ hai thu hẹp dần (H/2) theo thông lệ thiết kế hình phễu,
            # giúp nén biểu diễn trước khi phân loại thay vì giữ nguyên bề rộng.
            h2_dim = max(num_classes * 2, clf_hidden // 2)
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, clf_hidden), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(clf_hidden, h2_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2_dim, num_classes))
            self.lstm = None
        elif ct == "lstm":
            # chuỗi TẦNG: mỗi bước là biểu diễn của một tầng, chiếu về out_dim
            self.layer_proj = nn.ModuleList()
            wide_dim = hidden_dim * heads
            for li in range(self.num_layers):
                d_li = out_dim if li == self.num_layers - 1 else wide_dim
                self.layer_proj.append(
                    nn.Identity() if d_li == out_dim else nn.Linear(d_li, out_dim))
            self.lstm = nn.LSTM(out_dim, clf_hidden, batch_first=True)
            extra = clf_in - out_dim          # phần raw-skip (nếu bật)
            self.classifier = nn.Linear(clf_hidden + max(0, extra), num_classes)
        else:
            self.classifier = nn.Linear(clf_in, num_classes)
            self.lstm = None
        self._last_attn = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False,
                return_embeddings=False, return_feat=False):
        """return_feat=True: trả (logits, z) với z là biểu diễn ẩn TRƯỚC bộ
        phân loại, CÒN GIỮ đồ thị tính toán. Khác return_embeddings ở chỗ đó:
        return_embeddings gọi .detach() nên chỉ dùng để trực quan hoá, không
        lan truyền ngược được. ATA cần gradient đi qua z nên phải có nhánh này."""
        ea = edge_attr if self.use_edge else None

        h0 = self.dyt_in(self.input_proj(x))
        h0 = F.elu(h0)

        h = h0
        h_first = None                      # lưu đầu ra tầng 1 cho nhánh JK
        per_layer = []                      # đầu ra TỪNG tầng (phục vụ t-SNE)
        attn = None
        L = self.num_layers
        for li, (gat, nrm) in enumerate(zip(self.gats, self.norms)):
            last = (li == L - 1)
            prev = h
            if self.conv_type == "sage":
                h = gat(h, edge_index)          # không chú ý, không edge_attr
            elif return_attention and last:
                h, attn = gat(h, edge_index, edge_attr=ea,
                              return_attention_weights=True)
            else:
                h = gat(h, edge_index, edge_attr=ea)
            h = nrm(h)
            if not last:
                h = F.elu(h)
                # phần dư từng tầng cho các tầng GIỮA (kích thước đã khớp)
                if (self.mid_res is not None and 1 <= li <= L - 2
                        and prev.shape == h.shape):
                    h = h + prev
                h = F.dropout(h, p=self.dropout, training=self.training)
            if li == 0:
                h_first = h
            per_layer.append(h.detach())
        if return_attention:
            self._last_attn = attn
        x1 = h_first if h_first is not None else h0
        x2 = h

        # Residual chính: cộng biểu diễn input đã chiếu về out_dim
        if self.use_main_residual:
            x2 = F.elu(x2 + self.res_proj(h0))
        else:
            x2 = F.elu(x2)          # giữ phi tuyến để hai nhánh so sánh công bằng

        h2 = x2                                   # biểu diễn sau tầng GAT 2

        # ── Ghép các nhánh skip ────────────────────────────────────────
        parts = [h2]
        if self.skip_mode == "hybrid":
            g = torch.sigmoid(self.jk_gate)        # cổng học được ∈ (0,1)
            parts.append(g * self.jk_proj(x1))     # nhánh JK từ tầng 1
        if self.use_input_skip:
            parts.append(x)                        # raw-skip
        z = torch.cat(parts, dim=-1) if len(parts) > 1 else h2

        if self.classifier_type == "lstm":
            # chuỗi tầng [n, L, out_dim] -> LSTM -> trạng thái ẩn cuối
            seq = torch.stack([pj(t) for pj, t in zip(self.layer_proj, per_layer)], dim=1)
            _, (h_n, _) = self.lstm(seq)
            feat = h_n[-1]                                  # [n, clf_hidden]
            if z.shape[1] > x2.shape[1]:                    # còn phần raw-skip
                feat = torch.cat([feat, z[:, x2.shape[1]:]], dim=-1)
            logits = self.classifier(feat)
        else:
            logits = self.classifier(z)

        if return_feat:
            return logits, z
        if return_embeddings:
            # phục vụ trực quan hoá t-SNE theo từng tầng
            out = {"h%d" % (i + 1): t for i, t in enumerate(per_layer)}
            out["final"] = z.detach()
            out["logits"] = logits.detach()
            return out
        if return_attention:
            return logits, self._last_attn
        return logits

    @torch.no_grad()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict, meta: dict) -> EdgeAwareGAT:
    m = cfg["model"]
    model = EdgeAwareGAT(
        in_dim=meta["n_node_features"],
        edge_dim=meta["n_edge_features"],
        hidden_dim=m["hidden_dim"],
        out_dim=m["out_dim"],
        num_classes=meta["num_classes"],
        heads=m["heads"],
        dropout=m["dropout"],
        use_edge_features=m["use_edge_features"],
        use_dyt=m.get("use_dyt", True),
        use_input_skip=m.get("use_input_skip", True),
        skip_mode=m.get("skip_mode", "raw"),
        num_layers=m.get("num_layers", 2),
        layer_residual=m.get("layer_residual", False),
        use_main_residual=m.get("use_main_residual", True),
        classifier_type=m.get("classifier_type", "linear"),
        clf_hidden=m.get("clf_hidden", 64),
        conv_type=m.get("conv_type", "gatv2"),
    )
    return model
