#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attention_per_class.py — MÔ HÌNH THỰC SỰ DÙNG GÌ ĐỂ PHÁT HIỆN TỪNG LOẠI TẤN CÔNG
================================================================================

Khác biệt với feature_importance_per_class.py
---------------------------------------------
    feature_importance_per_class.py  ->  "đặc trưng X mang nhiều THÔNG TIN
                                          phân biệt lớp Y trong DỮ LIỆU"
                                          (đo bằng rừng ngẫu nhiên, không
                                           liên quan tới EdgeGAT)

    tệp này                          ->  "EdgeGAT thực sự DỰA VÀO X để quyết
                                          định lớp Y"
                                          (đọc trực tiếp từ trọng số chú ý và
                                           mặt nạ GNNExplainer của CHÍNH mô
                                           hình đã huấn luyện)

Hai câu trả lời có thể KHÁC nhau, và chỗ khác nhau chính là phần đáng viết:
dữ liệu chứa tín hiệu không có nghĩa mô hình dùng tín hiệu đó.

Ba phân tích
------------
1. PHÂN BỐ CHÚ Ý THEO LỚP.  Với mỗi lớp c, lấy mọi cạnh trỏ vào nút thuộc lớp
   c và so sánh phân bố hệ số chú ý α với phân bố trên cạnh trỏ vào nút Normal.
   Nếu mô hình dùng cấu trúc để phát hiện lớp c thì hai phân bố phải khác nhau
   rõ rệt; nếu trùng nhau, mô hình đang quyết định chủ yếu bằng đặc trưng nút.

2. ĐẶC TRƯNG CẠNH NÀO ĐIỀU KHIỂN CHÚ Ý.  Tính tương quan Spearman giữa từng
   cột đặc trưng cạnh và hệ số α trên các cạnh quanh lớp c. Đây là bằng chứng
   TRỰC TIẾP nhất cho luận điểm của luận văn: nó chỉ ra đặc trưng cạnh nào
   thực sự đang lái trọng số chú ý, chứ không chỉ nằm trong dữ liệu.

3. GNNExplainer THEO LỚP.  Với một số nút đại diện của mỗi lớp, chạy
   GNNExplainer để lấy mặt nạ tầm quan trọng trên ĐẶC TRƯNG NÚT và trên CẠNH,
   rồi lấy trung bình theo lớp. GNNExplainer tối ưu một mặt nạ sao cho giữ
   được dự đoán của mô hình khi chỉ giữ lại phần được che — nên kết quả nói về
   chính mô hình, không phải về dữ liệu.

Cách dùng
---------
    python scripts/attention_per_class.py --data-dir data/processed \\
           --ckpt-dir ck_mlp --top 6 --out attention_per_class.csv

    # bỏ qua GNNExplainer nếu muốn chạy nhanh
    python scripts/attention_per_class.py --data-dir data/processed \\
           --ckpt-dir ck_mlp --no-explainer

Đọc kết quả
-----------
Cột "Δ chú ý" là chênh lệch α trung bình giữa cạnh quanh lớp c và cạnh quanh
lớp Normal, chuẩn hoá theo độ lệch chuẩn gộp (hệ số d của Cohen). Giá trị nhỏ
KHÔNG có nghĩa mô hình sai — nó có nghĩa mô hình phát hiện lớp đó chủ yếu bằng
đặc trưng nút chứ không bằng cấu trúc lân cận, và đó cũng là một phát hiện.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils.common import get_device, get_logger, set_seed   # noqa: E402

log = get_logger()


def load_model(ckpt_dir, device):
    """Nạp EdgeGAT đã huấn luyện từ thư mục checkpoint."""
    from src.models.edge_gat import EdgeAwareGAT
    mc_path = os.path.join(ckpt_dir, "model_config.json")
    w_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(mc_path):
        for cand in ("EdgeGAT", "."):
            p = os.path.join(ckpt_dir, cand, "model_config.json")
            if os.path.exists(p):
                mc_path = p
                w_path = os.path.join(os.path.dirname(p), "best_model.pt")
                break
    mc = json.load(open(mc_path, encoding="utf-8"))
    # model_config.json chứa cả khoá không thuộc chữ ký khởi tạo (model_type,
    # seed, ...). Lọc theo chữ ký thật để tránh vỡ khi format checkpoint đổi.
    import inspect
    ok = set(inspect.signature(EdgeAwareGAT.__init__).parameters) - {"self"}
    kw = {k: v for k, v in mc.items() if k in ok}
    model = EdgeAwareGAT(**kw)
    model.load_state_dict(torch.load(w_path, map_location=device))
    return model.to(device).eval(), kw


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb, na, nb = np.var(a, ddof=1), np.var(b, ddof=1), len(a), len(b)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return 0.0 if sp < 1e-12 else float((np.mean(a) - np.mean(b)) / sp)


def collect_attention(model, graphs, device, max_graphs=400):
    """Trả về (alpha, nhãn nút đích, đặc trưng cạnh) gộp trên nhiều đồ thị.

    SỬA LỖI (quan trọng): bản đầu lấy `graphs[:max_graphs]`, tức 400 đồ thị
    ĐẦU TIÊN của tập kiểm thử. Tập kiểm thử được sắp theo tệp mô phỏng, nên
    400 đồ thị đầu có thể đến trọn vẹn từ các tệp KHÔNG TẤN CÔNG. Khi đó mọi
    nút đều mang nhãn Normal, không lớp nào đủ 20 cạnh, và Phân tích 1+2 im
    lặng hoàn toàn — đúng hiện tượng quan sát được trong lần chạy đầu
    (n = 62950 cạnh, tất cả đều thuộc lớp Normal).

    Bản này LẤY MẪU NGẪU NHIÊN trên toàn tập kiểm thử, và ưu tiên các đồ thị
    CÓ chứa ít nhất một nút tấn công để mỗi lớp đủ mẫu thống kê.
    """
    rng = np.random.default_rng(42)
    has_atk = [i for i, g in enumerate(graphs) if int((g.y > 0).sum()) > 0]
    no_atk = [i for i, g in enumerate(graphs) if int((g.y > 0).sum()) == 0]
    n_atk = min(len(has_atk), int(max_graphs * 0.75))
    n_norm = min(len(no_atk), max_graphs - n_atk)
    pick = ([has_atk[i] for i in rng.permutation(len(has_atk))[:n_atk]]
            + [no_atk[i] for i in rng.permutation(len(no_atk))[:n_norm]])
    log.info("Lấy mẫu %d đồ thị (%d có tấn công, %d toàn Normal) trong tổng %d",
             len(pick), n_atk, n_norm, len(graphs))
    sel = [graphs[i] for i in pick]

    A, Y, E = [], [], []
    with torch.no_grad():
        for g in sel:
            g = g.to(device)
            out = model(g.x, g.edge_index, g.edge_attr, return_attention=True)
            if not (isinstance(out, tuple) and len(out) == 2):
                return None
            _, attn = out
            if attn is None:
                return None
            ei, alpha = attn
            alpha = alpha.mean(1) if alpha.dim() > 1 else alpha
            dst = ei[1].cpu().numpy()
            lab = g.y.cpu().numpy()
            m = dst < len(lab)
            A.append(alpha.cpu().numpy()[m])
            Y.append(lab[dst[m]])
            # cạnh tự vòng do add_self_loops thêm vào KHÔNG có edge_attr tương
            # ứng; chỉ ghép đặc trưng cho phần cạnh gốc
            ne = g.edge_attr.shape[0]
            ea = np.zeros((m.sum(), g.edge_attr.shape[1]), dtype=np.float32)
            k = min(ne, m.sum())
            ea[:k] = g.edge_attr.cpu().numpy()[:k]
            E.append(ea)
    if not A:
        return None
    return np.concatenate(A), np.concatenate(Y), np.concatenate(E)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 10 or np.std(x) < 1e-12:
        return 0.0
    r = spearmanr(x, y).statistic
    return 0.0 if np.isnan(r) else float(r)


def run_explainer(model, graphs, device, cls, cls_names, n_nodes, epochs):
    """GNNExplainer: mặt nạ tầm quan trọng ĐẶC TRƯNG NÚT, trung bình theo lớp."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        from torch_geometric.explain.config import ModelConfig
    except Exception as e:
        log.warning("Không nạp được GNNExplainer: %s", e)
        return None

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, edge_index, edge_attr=None, **kw):
            return self.m(x, edge_index, edge_attr)

    expl = Explainer(
        model=Wrap(model), algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model", node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification",
                                 task_level="node", return_type="raw"))
    masks, done = [], 0
    for g in graphs:
        idxs = (g.y == cls).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue
        g = g.to(device)
        for i in idxs[:2].tolist():
            try:
                e = expl(g.x, g.edge_index, edge_attr=g.edge_attr, index=int(i))
                masks.append(e.node_mask[int(i)].detach().cpu().numpy())
                done += 1
            except Exception:
                continue
            if done >= n_nodes:
                break
        if done >= n_nodes:
            break
    if not masks:
        return None
    return np.mean(np.abs(np.stack(masks)), axis=0), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints/EdgeGAT")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--max-graphs", type=int, default=400)
    ap.add_argument("--explainer-nodes", type=int, default=12)
    ap.add_argument("--explainer-epochs", type=int, default=80)
    ap.add_argument("--no-explainer", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--export-model-csv", default=None,
                    help="xuất CSV (lop, kind, feature, score) để vẽ biểu đồ "
                         "bằng scripts/xai_simple.py")
    a = ap.parse_args()

    set_seed(42)
    device = get_device("auto")
    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    cls_names = meta["class_names"]
    dataset = meta.get("dataset", "?")
    test = torch.load(os.path.join(a.data_dir, "test.pt"), weights_only=False)
    model, mc = load_model(a.ckpt_dir, device)

    if not mc.get("use_edge_features", True):
        log.warning("Checkpoint này TẮT đặc trưng cạnh — phân tích 2 sẽ vô nghĩa.")
    if str(mc.get("conv_type", "gatv2")).lower() == "sage":
        log.error("Checkpoint dùng SAGEConv: KHÔNG có hệ số chú ý để phân tích.")
        return 1

    # tên đặc trưng
    try:
        from scripts.feature_importance_per_class import load_names
        nn_names = load_names(dataset, meta["n_node_features"], 0, False)
        ne_names = load_names(dataset, 0, meta["n_edge_features"], True)
    except Exception:
        nn_names = [f"f{i}" for i in range(meta["n_node_features"])]
        ne_names = [f"e{i}" for i in range(meta["n_edge_features"])]

    got = collect_attention(model, test, device, a.max_graphs)
    if got is None:
        log.error("Không lấy được trọng số chú ý từ mô hình.")
        return 1
    alpha, ylab, eattr = got
    log.info("Thu được %d cạnh có hệ số chú ý từ %d đồ thị",
             len(alpha), min(len(test), a.max_graphs))

    a_norm = alpha[ylab == 0]
    rows = []
    model_rows = []   # (lop, kind, feature, score) cho xai_simple.py

    print("\n" + "═" * 78)
    print("PHÂN TÍCH 1 + 2 — CHÚ Ý QUANH TỪNG LỚP VÀ ĐẶC TRƯNG CẠNH LÁI CHÚ Ý")
    print("═" * 78)
    print(f"α trung bình quanh nút Normal: {a_norm.mean():.4f} "
          f"(độ lệch chuẩn {a_norm.std():.4f}, n = {len(a_norm)})")

    for c, cname in enumerate(cls_names):
        if c == 0:
            continue
        m = (ylab == c)
        if m.sum() < 20:
            continue
        a_c = alpha[m]
        dd = cohens_d(a_c, a_norm)
        print("\n" + "─" * 78)
        print(f"{cname}   α = {a_c.mean():.4f}   Δ so với Normal = {dd:+.3f} "
              f"(hệ số d)   n = {int(m.sum())}")
        if abs(dd) < 0.2:
            print("   Chú ý quanh lớp này gần như KHÔNG khác Normal ⟹ mô hình")
            print("   phát hiện nó chủ yếu bằng ĐẶC TRƯNG NÚT, không bằng cấu trúc.")
        cors = [(ne_names[j], spearman(eattr[m][:, j], a_c))
                for j in range(eattr.shape[1])]
        cors.sort(key=lambda t: -abs(t[1]))
        print(f"   Đặc trưng cạnh lái chú ý mạnh nhất (Spearman ρ với α):")
        for nm, r in cors[:min(4, a.top)]:
            bar = "▇" * int(abs(r) * 20)
            print(f"     {nm:<18}{r:+.3f}  {bar}")
        for nm, r in cors:
            model_rows.append(dict(lop=cname, kind="edge", feature=nm,
                                   score=abs(float(r))))
        rows.append(dict(lop=cname, alpha_mean=round(float(a_c.mean()), 5),
                         delta_vs_normal=round(dd, 4), n_edges=int(m.sum()),
                         top_edge_feat=cors[0][0], rho=round(cors[0][1], 4)))

    if not a.no_explainer:
        print("\n" + "═" * 78)
        print("PHÂN TÍCH 3 — GNNExplainer: ĐẶC TRƯNG NÚT MÔ HÌNH DỰA VÀO")
        print("═" * 78)
        for c, cname in enumerate(cls_names):
            if c == 0:
                continue
            r = run_explainer(model, test, device, c, cls_names,
                              a.explainer_nodes, a.explainer_epochs)
            if r is None:
                print(f"\n{cname}: không giải thích được (thiếu mẫu hoặc lỗi)")
                continue
            mask, n = r
            for j in range(len(mask)):
                model_rows.append(dict(lop=cname, kind="node",
                                       feature=nn_names[j],
                                       score=float(mask[j])))
            order = np.argsort(-mask)[:a.top]
            print(f"\n{cname}  (trung bình trên {n} nút)")
            for j in order:
                bar = "▇" * int(mask[j] / (mask.max() + 1e-12) * 22)
                print(f"   {nn_names[j]:<22}{mask[j]:.4f}  {bar}")

    print("\n" + "═" * 78)
    print("CÁCH VIẾT VÀO LUẬN VĂN")
    print("  Kết quả tệp này CHO PHÉP viết: 'mô hình dựa vào X để phát hiện Y'.")
    print("  Đối chiếu với bảng của feature_importance_per_class.py:")
    print("   · TRÙNG  ⟹ mô hình khai thác đúng tín hiệu mạnh nhất trong dữ liệu.")
    print("   · LỆCH   ⟹ phát hiện đáng viết: mô hình tìm được tín hiệu khác,")
    print("               hoặc bỏ lỡ tín hiệu mà dữ liệu có sẵn.")
    print("  |d| và |ρ| dưới 0,2 nên đọc là 'không đáng kể', đừng diễn giải quá.")

    if a.export_model_csv and model_rows:
        import csv
        with open(a.export_model_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["lop", "kind", "feature", "score"])
            w.writeheader(); w.writerows(model_rows)
        print(f"\nĐã ghi phía MÔ HÌNH -> {a.export_model_csv}")
        print("Bước tiếp: python scripts/xai_simple.py --model-csv "
              f"{a.export_model_csv} --rf-node-csv dactrung_node.csv "
              "--rf-edge-csv dactrung_canh.csv --out-dir hinh_xai")

    if a.out and rows:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nĐã ghi -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
