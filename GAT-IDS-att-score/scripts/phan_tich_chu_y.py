# -*- coding: utf-8 -*-
"""
phan_tich_chu_y.py — Phân tích TRỌNG SỐ CHÚ Ý của NE-GAT-IDS (v3.48)
=====================================================================

Trả lời ba câu hỏi, theo đúng thứ tự cần thiết để một nhận xét về chú ý
đứng vững trước hội đồng:

 (1) CHÚ Ý CÓ CHỌN LỌC KHÔNG?   Entropy chuẩn hoá của β quanh mỗi nút.
     H_i / log(d_i + 1) ≈ 1  -> β gần đều, mô hình KHÔNG phân biệt láng giềng.
 (2) CHÚ Ý DỒN VÀO ĐÂU?         So β của cạnh nối tới láng giềng TẤN CÔNG với
     cạnh nối tới láng giềng BÌNH THƯỜNG, ghép cặp trên cùng một nút trung tâm
     (kiểm định dấu hạng Wilcoxon). Kèm tương quan Spearman giữa β và từng
     đặc trưng cạnh gốc: đặc trưng cạnh nào điều khiển chú ý.
 (3) CHÚ Ý CÓ "TRUNG THỰC" KHÔNG? Phép thử xoá cạnh: xoá q% cạnh có β cao nhất,
     so với xoá q% cạnh NGẪU NHIÊN và q% cạnh có β THẤP nhất. Nếu β phản ánh
     điều mô hình thực sự dùng, xoá cạnh β cao phải làm hỏng dự đoán nhiều
     hơn hẳn hai phép còn lại (Jain & Wallace 2019; Serrano & Smith 2019).

Chỉ áp cho checkpoint NE-GAT do train_ata.py sinh ra (cần model_config.json
và best_model.pt). Ví dụ:

  python scripts/phan_tich_chu_y.py --data-dir data/radar_chrono ^
      --ckpt-dir ck_negat_ata_radar_seed42 --out-dir phan_tich_chu_y/radar_s42

Kết quả: tom_tat.json, beta_canh.csv (mỗi cạnh một dòng), ba hình PNG.
"""
import argparse, json, os, sys, math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.models.ne_gat import NEGAT  # noqa: E402


def load_model(ckpt_dir, device):
    mc = json.load(open(os.path.join(ckpt_dir, "model_config.json"), encoding="utf-8"))
    m = NEGAT(in_dim=mc["in_dim"], edge_dim=mc["edge_dim"], num_classes=mc["num_classes"],
              hidden_dim=mc["hidden_dim"], out_dim=mc["out_dim"],
              edge_latent=mc.get("edge_latent", 24), node_heads=mc.get("node_heads", 2),
              num_layers=mc.get("num_layers", 3), dropout=mc["dropout"],
              classifier_type=mc.get("classifier_type", "mlp"), clf_hidden=mc.get("clf_hidden", 128),
              edge_score_mode=mc.get("edge_score_mode", "full"), edge_agg=mc.get("edge_agg", "mean_deg"),
              edge_update=mc.get("edge_update", True), use_dyt=mc.get("use_dyt", True), verbose=False)
    sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device, weights_only=True)
    miss, unexp = m.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit("Trọng số không khớp kiến trúc (thiếu %d, thừa %d khoá) — sai checkpoint?"
                         % (len(miss), len(unexp)))
    m.to(device).eval()
    for l in m.layers:
        l.edge.keep_att = True
        l.node.keep_att = True
    return m, mc


def macro_f1(y, p, K):
    f = []
    for c in range(K):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        if tp + fn == 0:
            continue
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


@torch.no_grad()
def run_graph(m, g, device, keep=None):
    ei, ea = g.edge_index, g.edge_attr
    if keep is not None:
        ei, ea = ei[:, keep], ea[keep]
    logits = m(g.x.to(device), ei.to(device), ea.to(device))
    return torch.softmax(logits, -1).cpu()


# meta.json của các bộ đã chuyển đổi không lưu tên đặc trưng cạnh; chép đúng thứ tự từ
# radar_builder.py (RADAR_EDGE_FEATURE_NAMES), rplguard_builder.py (EDGE_FEATURE_NAMES)
# và rplbeh_builder.py (EDGE_FEAT_NAMES). Chỉ dùng khi số lượng khớp n_edge_features.
TEN_CANH = {
    "radar": ["count", "log_count", "pay_avg", "dio_ratio", "dao_ratio",
              "dis_ratio", "data_ratio", "col_ratio", "is_parent_edge", "dir_flag"],
    "rplguard": ["pkt_count", "etx_mean", "hops_mean", "delta_time_mean",
                 "distance_mean", "drop_diff", "fwd_ratio", "energy_diff", "is_reverse"],
    "rplbeh": ["abs_rank_diff", "signed_rank_gap", "abs_ver_diff", "ctrl_vol_parent", "ctrl_vol_child",
               "fwd_to_me_child", "fwd_ratio_gap", "dis_rate_gap", "n_siblings", "direction"],
}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", default="phan_tich_chu_y")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--max-graphs", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--n-del", type=int, default=600, help="số đồ thị dùng cho phép thử xoá cạnh")
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.5")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cpu")
    meta = json.load(open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8"))
    names = meta.get("class_names", [str(i) for i in range(meta["num_classes"])])
    efn = meta.get("edge_feats") or meta.get("edge_feature_names") or TEN_CANH.get(meta.get("dataset"))
    if not efn or len(efn) != meta["n_edge_features"]:
        efn = ["e%d" % i for i in range(meta["n_edge_features"])]
    gs = torch.load(os.path.join(a.data_dir, a.split + ".pt"), weights_only=False)
    if a.max_graphs and a.max_graphs < len(gs):
        gs = [gs[i] for i in sorted(rng.choice(len(gs), a.max_graphs, replace=False))]
    m, mc = load_model(a.ckpt_dir, dev)
    L = len(m.layers); K = meta["num_classes"]
    print("Mô hình NE-GAT | edge_score=%s | %d tầng | %d đồ thị %s" % (mc.get("edge_score_mode"), L, len(gs), a.split))

    # ───────────── (1) + (2): thu β, α theo từng cạnh ─────────────
    rows = []          # mỗi cạnh thật: g, tầng, src, dst, y_src, y_dst, beta, alpha, deg, feats
    ent = {l: [] for l in range(L)}
    pair = {l: [] for l in range(L)}   # (β̄ láng giềng tấn công − β̄ láng giềng thường) theo nút trung tâm
    yt_all, yp_all = [], []
    for gi, g in enumerate(gs):
        prob = run_graph(m, g, dev)
        yt_all.append(g.y.numpy()); yp_all.append(prob.argmax(-1).numpy())
        y = g.y.numpy(); E = g.edge_index.size(1); N = g.x.size(0)
        ea = g.edge_attr.numpy()
        for l, layer in enumerate(m.layers):
            src, dst, b = [t.numpy() for t in layer.edge.last_att]
            nsrc, ndst, na = [t.numpy() for t in layer.node.last_att]
            # kênh cạnh: E cạnh thật nằm trước, N tự vòng nằm sau
            bs, bd, bb = src[:E], dst[:E], b[:E]
            deg = np.bincount(bd, minlength=N)
            # α kênh nút: GATv2Conv tự thêm tự vòng; ghép theo (src,dst) cho cạnh thật
            amap = {}
            for s_, d_, v_ in zip(nsrc, ndst, na):
                if s_ != d_:
                    amap[(int(s_), int(d_))] = float(v_)
            for k in range(E):
                rows.append((gi, l, int(bs[k]), int(bd[k]), int(y[bs[k]]), int(y[bd[k]]), float(bb[k]),
                             amap.get((int(bs[k]), int(bd[k])), float("nan")), int(deg[bd[k]]), *ea[k].tolist()))
            # entropy chuẩn hoá theo nút (gồm cả tự vòng, đúng phân phối softmax)
            for i in range(N):
                w = b[dst == i]
                if w.size >= 3:   # ít nhất 2 láng giềng thật + tự vòng
                    w = np.clip(w, 1e-12, 1)
                    ent[l].append(float(-(w * np.log(w)).sum() / math.log(w.size)))
                msk = bd == i
                if msk.sum() >= 2:
                    ys = y[bs[msk]]; ww = bb[msk]
                    if (ys != 0).any() and (ys == 0).any():
                        pair[l].append((int(y[i]), float(ww[ys != 0].mean() - ww[ys == 0].mean()),
                                        float(ww[ys != 0].mean() * (msk.sum() + 1)), float(ww[ys == 0].mean() * (msk.sum() + 1))))
        if (gi + 1) % 500 == 0:
            print("  đã xử lý %d/%d đồ thị" % (gi + 1, len(gs)), flush=True)

    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    mf1 = macro_f1(yt, yp, K)
    cols = ["graph", "layer", "src", "dst", "y_src", "y_dst", "beta", "alpha_node", "deg_dst"] + list(efn)
    import csv
    with open(os.path.join(a.out_dir, "beta_canh.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    R = np.array([r[:9] for r in rows], dtype=float)
    F = np.array([r[9:] for r in rows], dtype=float)

    from scipy import stats
    out = {"edge_score_mode": mc.get("edge_score_mode"), "macro_f1_kiem_tra": mf1, "so_do_thi": len(gs), "tang": []}
    for l in range(L):
        sel = R[:, 1] == l
        beta = R[sel, 6]; degd = R[sel, 8]; ratio = beta * (degd + 1)   # 1 = đúng bằng mức đều
        p = np.array(pair[l]) if pair[l] else np.zeros((0, 4))
        rec = {"tang": l + 1,
               "entropy_chuan_hoa_tb": float(np.mean(ent[l])) if ent[l] else None,
               "ti_le_nut_entropy_tren_0.95": float(np.mean(np.array(ent[l]) > 0.95)) if ent[l] else None,
               "so_nut_co_ca_hai_loai_lang_gieng": int(len(p))}
        if len(p) >= 10:
            d = p[:, 1]
            wres = stats.wilcoxon(d) if np.any(d != 0) else None
            rec.update({"chenh_beta_tan_cong_tru_thuong_trung_vi": float(np.median(d)),
                        "ti_le_nut_beta_tan_cong_lon_hon": float(np.mean(d > 0)),
                        "wilcoxon_p": float(wres.pvalue) if wres else None,
                        "ti_so_so_voi_deu_lang_gieng_tan_cong": float(np.mean(p[:, 2])),
                        "ti_so_so_voi_deu_lang_gieng_thuong": float(np.mean(p[:, 3]))})
            by = {}
            for c in np.unique(p[:, 0]).astype(int):
                dd = d[p[:, 0] == c]
                if len(dd) >= 10:
                    by[names[c]] = {"n": int(len(dd)), "trung_vi": float(np.median(dd)), "ti_le_duong": float(np.mean(dd > 0))}
            rec["theo_lop_nut_trung_tam"] = by
        cor = {}
        for j, nm in enumerate(efn):
            if np.std(F[sel, j]) > 0:
                rho, pv = stats.spearmanr(ratio, F[sel, j])
                cor[nm] = {"rho": float(rho), "p": float(pv)}
        rec["spearman_beta_chuan_hoa_vs_dac_trung_canh"] = dict(sorted(cor.items(), key=lambda kv: -abs(kv[1]["rho"])))
        an = R[sel, 7]
        if np.isfinite(an).sum() > 10:
            ok = np.isfinite(an)
            rec["spearman_beta_canh_vs_alpha_nut"] = float(stats.spearmanr(beta[ok], an[ok])[0])
        out["tang"].append(rec)

    # ───────────── (3) phép thử xoá cạnh ─────────────
    fracs = [float(x) for x in a.fracs.split(",")]
    idx = rng.choice(len(gs), min(a.n_del, len(gs)), replace=False)
    score = {}   # điểm của cạnh = β trung bình qua các tầng (kênh cạnh)
    for gi in idx:
        sel = R[:, 0] == gi
        E = gs[gi].edge_index.size(1)
        if E == 0:
            continue
        score[gi] = R[sel, 6].reshape(L, E).mean(0)
    def eval_keep(rule, q, rep=0):
        yt_, yp_, dp = [], [], []
        r2 = np.random.default_rng(1000 + rep)
        for gi in score:
            g = gs[gi]; s = score[gi]; E = s.size; nk = int(round(q * E))
            if rule == "top":
                drop = np.argsort(-s)[:nk]
            elif rule == "bottom":
                drop = np.argsort(s)[:nk]
            else:
                drop = r2.choice(E, nk, replace=False)
            keep = np.setdiff1d(np.arange(E), drop)
            p0 = run_graph(m, g, dev); p1 = run_graph(m, g, dev, torch.as_tensor(keep, dtype=torch.long))
            c0 = p0.argmax(-1)
            dp.append((p0.gather(1, c0[:, None]) - p1.gather(1, c0[:, None])).squeeze(1).numpy())
            yt_.append(g.y.numpy()); yp_.append(p1.argmax(-1).numpy())
        yt_ = np.concatenate(yt_); yp_ = np.concatenate(yp_); dp = np.concatenate(dp)
        att = yt_ != 0
        return macro_f1(yt_, yp_, K), float(dp.mean()), float(dp[att].mean()) if att.any() else float("nan")
    base_y = np.concatenate([gs[gi].y.numpy() for gi in score])
    base_p = np.concatenate([run_graph(m, gs[gi], dev).argmax(-1).numpy() for gi in score])
    dl = {"macro_f1_goc": macro_f1(base_y, base_p, K), "ket_qua": []}
    for q in fracs:
        t = eval_keep("top", q); b = eval_keep("bottom", q)
        rr = [eval_keep("random", q, k) for k in range(a.n_random)]
        dl["ket_qua"].append({"q": q, "xoa_beta_cao": {"macro_f1": t[0], "giam_xs_tb": t[1], "giam_xs_nut_tan_cong": t[2]},
                              "xoa_ngau_nhien": {"macro_f1": float(np.mean([x[0] for x in rr])), "sd": float(np.std([x[0] for x in rr], ddof=1)) if len(rr) > 1 else 0.0,
                                                  "giam_xs_tb": float(np.mean([x[1] for x in rr])), "giam_xs_nut_tan_cong": float(np.mean([x[2] for x in rr]))},
                              "xoa_beta_thap": {"macro_f1": b[0], "giam_xs_tb": b[1], "giam_xs_nut_tan_cong": b[2]}})
        print("  xoá %.0f%% cạnh: β cao -> mF1 %.4f | ngẫu nhiên -> %.4f | β thấp -> %.4f" % (
            100 * q, t[0], dl["ket_qua"][-1]["xoa_ngau_nhien"]["macro_f1"], b[0]), flush=True)
    out["phep_thu_xoa_canh"] = dl
    json.dump(out, open(os.path.join(a.out_dir, "tom_tat.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ───────────── hình ─────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for l in range(L):
        if ent[l]:
            axs[0].hist(ent[l], bins=30, alpha=0.5, label="tầng %d" % (l + 1))
    axs[0].set_title("Entropy chuẩn hoá của β (1 = đều)"); axs[0].legend()
    xs = np.arange(L); r1 = [out["tang"][l].get("ti_so_so_voi_deu_lang_gieng_tan_cong", np.nan) for l in range(L)]
    r0 = [out["tang"][l].get("ti_so_so_voi_deu_lang_gieng_thuong", np.nan) for l in range(L)]
    axs[1].bar(xs - 0.2, r1, 0.4, label="láng giềng tấn công"); axs[1].bar(xs + 0.2, r0, 0.4, label="láng giềng bình thường")
    axs[1].axhline(1, color="k", ls="--", lw=1); axs[1].set_xticks(xs); axs[1].set_xticklabels(["tầng %d" % (l + 1) for l in range(L)])
    axs[1].set_title("β so với mức chia đều (1 = đều)"); axs[1].legend()
    qq = [r["q"] * 100 for r in dl["ket_qua"]]
    axs[2].plot(qq, [r["xoa_beta_cao"]["macro_f1"] for r in dl["ket_qua"]], "o-", label="xoá cạnh β cao")
    axs[2].plot(qq, [r["xoa_ngau_nhien"]["macro_f1"] for r in dl["ket_qua"]], "s--", label="xoá ngẫu nhiên")
    axs[2].plot(qq, [r["xoa_beta_thap"]["macro_f1"] for r in dl["ket_qua"]], "^:", label="xoá cạnh β thấp")
    axs[2].axhline(dl["macro_f1_goc"], color="k", lw=1); axs[2].set_xlabel("% cạnh bị xoá"); axs[2].set_ylabel("macro-F1")
    axs[2].set_title("Phép thử xoá cạnh"); axs[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "phan_tich_chu_y.png"), dpi=200)
    print("Đã lưu ->", a.out_dir)


if __name__ == "__main__":
    main()
