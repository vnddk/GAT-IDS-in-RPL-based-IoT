# -*- coding: utf-8 -*-
"""
tong_hop_canh_chu_y.py — Kiểm định đóng góp của ĐẶC TRƯNG CẠNH TRONG HỆ SỐ CHÚ Ý
(v3.48). So sánh ghép cặp theo hạt giống giữa hai nhánh NE-GAT-IDS:

    A = edge_score "full"       β_ij = softmax(aᵀ LReLU(W[h_i ‖ h_j ‖ e_ij]))
    B = edge_score "node_only"  β_ij = softmax(aᵀ LReLU(W[h_i ‖ h_j]))
        (e_ij vẫn đi vào giá trị W_v·e_ij; mọi thành phần khác giữ nguyên)

Mỗi nhánh nhận một trong hai dạng đầu vào:
  --a-ckpt "ck_negat_full_radar_seed{S}"   -> nạp best_model.pt, TỰ ĐÁNH GIÁ lại
                                               trên tập kiểm thử (có F1 theo lớp)
  --a-values "0.9396,0.9482,0.9429,0.9317,0.9419" -> dùng macro-F1 đã ghi trong
                                               nhật ký (không có F1 theo lớp)

  python scripts/tong_hop_canh_chu_y.py --data-dir data/radar_chrono ^
      --a-ckpt "ck_negat_full_radar_seed{S}" --b-ckpt "ck_negat_nodeonly_radar_seed{S}" ^
      --seeds 42,43,44,45,46 --out-dir ket_qua_canh_chu_y/radar

Kết quả: tom_tat.json, theo_lop.csv, hieu_ghep_cap.png và một câu kết luận
viết sẵn theo đúng quy tắc quyết định trong hướng dẫn.
"""
import argparse, csv, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.models.ne_gat import NEGAT  # noqa: E402


def load(ckpt_dir):
    mc = json.load(open(os.path.join(ckpt_dir, "model_config.json"), encoding="utf-8"))
    m = NEGAT(in_dim=mc["in_dim"], edge_dim=mc["edge_dim"], num_classes=mc["num_classes"],
              hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"], edge_latent=mc.get("edge_latent", 24),
              node_heads=mc.get("node_heads", 2), num_layers=mc.get("num_layers", 3), dropout=mc["dropout"],
              classifier_type=mc.get("classifier_type", "mlp"), clf_hidden=mc.get("clf_hidden", 128),
              edge_score_mode=mc.get("edge_score_mode", "full"), edge_agg=mc.get("edge_agg", "mean_deg"),
              edge_update=mc.get("edge_update", True), use_dyt=mc.get("use_dyt", True), verbose=False)
    sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu", weights_only=True)
    miss, unexp = m.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit("%s: trọng số không khớp kiến trúc (thiếu %d, thừa %d khoá)" % (ckpt_dir, len(miss), len(unexp)))
    m.eval()
    npar = sum(p.numel() for p in m.parameters())
    use_edge = mc.get("use_edge_features", True)
    mode = mc.get("edge_score_mode", "full") + ("" if use_edge else "+no_edge")
    return m, mode, npar, use_edge


@torch.no_grad()
def evaluate(m, gs, K, use_edge=True):
    # Nhánh --no-edge được huấn luyện với edge_attr bị xoá trắng ở cả ba tập
    # (train_ata.py v3.48), nên khi đánh giá lại cũng phải xoá trắng như vậy.
    yt, yp = [], []
    for g in gs:
        ea = g.edge_attr if use_edge else torch.zeros_like(g.edge_attr)
        yt.append(g.y.numpy()); yp.append(m(g.x, g.edge_index, ea).argmax(-1).numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    f1 = {}
    for c in range(K):
        tp = int(((yp == c) & (yt == c)).sum()); fp = int(((yp == c) & (yt != c)).sum()); fn = int(((yp != c) & (yt == c)).sum())
        if tp + fn:
            f1[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.mean(list(f1.values()))), f1


def dz_luc80(n, luc=0.8):
    """Cỡ hiệu ứng d_z nhỏ nhất để kiểm định t ghép cặp hai phía (α = 0,05, n cặp) đạt lực 80 %.
    Quét lưới thay cho brentq vì scipy.stats.nct trả NaN khi tham số phi tâm lớn."""
    from scipy import stats
    tc = stats.t.ppf(0.975, n - 1)
    z = np.linspace(0.01, 8, 8000)
    nc = z * np.sqrt(n)
    pw = np.nan_to_num(stats.nct.sf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc), nan=1.0)
    return float(z[np.argmax(pw >= luc)])


def arm(name, pattern, values, seeds, gs, K):
    if values:
        v = [float(x) for x in values.split(",")]
        if len(v) != len(seeds):
            raise SystemExit("--%s-values cần đúng %d giá trị" % (name, len(seeds)))
        return {"mf1": v, "cls": None, "mode": "nhat_ky", "edge_score": None, "params": None}
    mf, cl, modes, npar = [], [], set(), set()
    for s in seeds:
        d = pattern.replace("{S}", str(s))
        m, mode, p, ue = load(d)
        a, f = evaluate(m, gs, K, ue)
        mf.append(a); cl.append(f); modes.add(mode); npar.add(p)
        print("  %s hạt giống %d: %s | edge_score=%s | %d tham số | macro-F1 = %.4f" % (name, s, d, mode, p, a), flush=True)
    if len(modes) > 1:
        raise SystemExit("Nhánh %s trộn nhiều edge_score khác nhau: %s" % (name, modes))
    return {"mf1": mf, "cls": cl, "mode": "checkpoint", "edge_score": modes.pop(), "params": npar.pop()}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--a-ckpt"); ap.add_argument("--a-values")
    ap.add_argument("--b-ckpt"); ap.add_argument("--b-values")
    ap.add_argument("--a-name", default="full"); ap.add_argument("--b-name", default="node_only")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out-dir", default="ket_qua_canh_chu_y")
    a = ap.parse_args()
    from scipy import stats
    os.makedirs(a.out_dir, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    K = meta["num_classes"]; names = meta.get("class_names", [str(i) for i in range(K)])
    gs = torch.load(os.path.join(a.data_dir, "test.pt"), weights_only=False)
    A = arm(a.a_name, a.a_ckpt, a.a_values, seeds, gs, K)
    B = arm(a.b_name, a.b_ckpt, a.b_values, seeds, gs, K)
    if A["edge_score"] and B["edge_score"] and A["edge_score"] == B["edge_score"]:
        raise SystemExit("Hai nhánh có CÙNG edge_score = %s — sai thư mục?" % A["edge_score"])
    x = np.array(A["mf1"]); y = np.array(B["mf1"]); d = x - y; n = len(d)
    sd = d.std(ddof=1); t = stats.t.ppf(0.975, n - 1); ci = t * sd / np.sqrt(n)
    tt = stats.ttest_rel(x, y)
    res = {"seeds": seeds, a.a_name: A["mf1"], a.b_name: B["mf1"],
           "tb_" + a.a_name: float(x.mean()), "sd_" + a.a_name: float(x.std(ddof=1)),
           "tb_" + a.b_name: float(y.mean()), "sd_" + a.b_name: float(y.std(ddof=1)),
           "hieu_tb": float(d.mean()), "hieu_sd": float(sd), "ktc95": [float(d.mean() - ci), float(d.mean() + ci)],
           "t": float(tt.statistic), "p": float(tt.pvalue), "d_z": float(d.mean() / sd) if sd > 0 else None,
           "so_hat_giong_A_thang": int((d > 0).sum()), "tham_so": {a.a_name: A["params"], a.b_name: B["params"]}}
    lo, hi = res["ktc95"]
    if lo > 0 and res["p"] < 0.05:
        kl = ("Đưa đặc trưng cạnh vào hệ số chú ý làm tăng macro-F1 %.4f điểm (KTC 95 %% [%.4f; %.4f], t(%d) = %.2f, "
              "p = %.4f, d_z = %.2f, thắng %d/%d hạt giống)." % (d.mean(), lo, hi, n - 1, tt.statistic, tt.pvalue, res["d_z"], (d > 0).sum(), n))
    elif hi < 0 and res["p"] < 0.05:
        kl = ("Đưa đặc trưng cạnh vào hệ số chú ý làm GIẢM macro-F1 %.4f điểm (KTC 95 %% [%.4f; %.4f], p = %.4f)." % (-d.mean(), lo, hi, tt.pvalue))
    else:
        dz80 = dz_luc80(n)
        kl = ("Chưa đủ bằng chứng: hiệu %.4f, KTC 95 %% [%.4f; %.4f] chứa 0 (p = %.3f). Với độ lệch chuẩn hiệu %.4f, %d hạt "
              "giống chỉ phát hiện tin cậy (lực 80 %%) chênh lệch từ khoảng %.4f trở lên." % (d.mean(), lo, hi, tt.pvalue, sd, n, dz80 * sd))
    res["ket_luan"] = kl
    print("\n%s: %.4f ± %.4f | %s: %.4f ± %.4f" % (a.a_name, x.mean(), x.std(ddof=1), a.b_name, y.mean(), y.std(ddof=1)))
    print("KẾT LUẬN:", kl)
    if A["cls"] and B["cls"]:
        rows = [["lop", "n_test", a.a_name + "_tb", a.b_name + "_tb", "hieu_tb", "hieu_sd", "thang_A"]]
        cnt = {}
        for g in gs:
            for c in g.y.numpy():
                cnt[int(c)] = cnt.get(int(c), 0) + 1
        for c in sorted(A["cls"][0]):
            fa = np.array([f[c] for f in A["cls"]]); fb = np.array([f[c] for f in B["cls"]]); dd = fa - fb
            rows.append([names[c], cnt.get(c, 0), "%.4f" % fa.mean(), "%.4f" % fb.mean(), "%+.4f" % dd.mean(),
                         "%.4f" % dd.std(ddof=1), "%d/%d" % ((dd > 0).sum(), n)])
        with open(os.path.join(a.out_dir, "theo_lop.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("\nF1 theo lớp:")
        for r in rows:
            print("  " + "  ".join("%-20s" % v if i == 0 else "%-10s" % v for i, v in enumerate(r)))
    json.dump(res, open(os.path.join(a.out_dir, "tom_tat.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(n):
        ax[0].plot([0, 1], [x[i], y[i]], "o-", color="#888")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels([a.a_name, a.b_name]); ax[0].set_ylabel("macro-F1")
    ax[0].set_title("Từng hạt giống (mỗi đường một hạt giống)")
    ax[1].bar(range(n), d, color=["#2a78d6" if v > 0 else "#e34948" for v in d]); ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(range(n)); ax[1].set_xticklabels([str(s) for s in seeds]); ax[1].set_title("Hiệu %s − %s" % (a.a_name, a.b_name))
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "hieu_ghep_cap.png"), dpi=200)
    print("Đã lưu ->", a.out_dir)


if __name__ == "__main__":
    main()
