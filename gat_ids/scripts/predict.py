"""BƯỚC 3 — Dự đoán trên mẫu MỚI bằng model ĐÃ TRAIN (< 1 giây, KHÔNG retrain).
Chạy: python scripts/predict.py --input new_traffic.csv --dataset radar
"""
import os, sys, json, argparse, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.common import get_device, get_logger
from src.data.loader import GraphScaler
from src.models.edge_gat import EdgeAwareGAT
log = get_logger()

def _load_model(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, "model_config.json")) as f: mc = json.load(f)
    m = EdgeAwareGAT(in_dim=mc["in_dim"],edge_dim=mc["edge_dim"],hidden_dim=mc["hidden_dim"],
                     out_dim=mc["out_dim"],num_classes=mc["num_classes"],heads=mc["heads"],
                     dropout=mc["dropout"],use_edge_features=mc["use_edge_features"],use_dyt=mc.get("use_dyt",True),
                     use_input_skip=mc.get("use_input_skip",True),skip_mode=mc.get("skip_mode","raw"),
            num_layers=mc.get("num_layers", 2),
            layer_residual=mc.get("layer_residual", False),
        use_main_residual=mc.get("use_main_residual", True),
        classifier_type=mc.get("classifier_type", "linear"),
        clf_hidden=mc.get("clf_hidden", 64))
    m.load_state_dict(torch.load(os.path.join(ckpt_dir,"best_model.pt"),map_location=device,weights_only=True))
    m.to(device).eval(); return m, mc

def _load_scaler(data_dir):
    """Nạp scaler ĐÚNG loại đã dùng khi train.

    SỬA HAI LỖI:
    (a) weights_only=True làm torch.load NÉM LỖI: scaler.pt chứa ndarray của
        numpy chứ không chỉ tensor. Với cấu hình mặc định (use_power: true)
        predict.py không chạy được lần nào.
    (b) Nghiêm trọng hơn: bản cũ luôn dựng GraphScaler (z-score) và chỉ lấy
        mean/std, BỎ QUA λ Yeo-Johnson. Nếu load có thành công thì mẫu mới vẫn
        bị chuẩn hoá khác với lúc train -> dự đoán sai một cách ÂM THẦM, không
        báo lỗi. Nay chọn lớp scaler theo cờ use_power đã lưu.
    """
    s = torch.load(os.path.join(data_dir, "scaler.pt"), weights_only=False)
    if s.get("use_power", False):
        from src.data.transforms import GraphPowerScaler
        sc = GraphPowerScaler(use_power=True)
        sc.x_lam, sc.e_lam = s["x_lam"], s["e_lam"]
    else:
        sc = GraphScaler()
    sc.x_mean, sc.x_std, sc.e_mean, sc.e_std = s["x_mean"], s["x_std"], s["e_mean"], s["e_std"]
    return sc

def _convert(csv_path, dataset, ws):
    import pandas as pd; df = pd.read_csv(csv_path)
    if dataset == "uos":
        from src.data.uos_builder import build_graphs_from_df; return build_graphs_from_df(df, ws)
    elif dataset == "radar":
        from src.data.radar_builder import (build_graphs_from_radar_csv,
                                             CLASS_TO_IDX, _parse_attacks_start_time)
        # Tên attack: thư mục cha, bỏ qua tầng Packet_Trace_1500s nếu có
        d = os.path.dirname(os.path.abspath(csv_path))
        p = os.path.basename(d).lower()
        if p.startswith("packet_trace"):
            p = os.path.basename(os.path.dirname(d)).lower()
        at = p if p in CLASS_TO_IDX else "Normal"
        # Hybrid labeling tầng 2: đọc attacks_start_time.txt nếu có
        attacker, start_s = None, -1.0
        for td in (d, os.path.dirname(d)):
            txt = os.path.join(td, "attacks_start_time.txt")
            if os.path.isfile(txt):
                info = _parse_attacks_start_time(txt)
                try: sim_id = int(os.path.splitext(os.path.basename(csv_path))[0])
                except ValueError: sim_id = -1
                if sim_id in info: attacker, start_s = info[sim_id]
                break
        gs, mode = build_graphs_from_radar_csv(df, at, ws,
                                                attacker_node=attacker,
                                                attack_start_s=start_s)
        log.info("Label mode: %s (attacker=%s)", mode, attacker or "N/A")
        return gs
    raise ValueError(f"Dataset '{dataset}' chưa có builder")

@torch.no_grad()
def predict(model, graphs, device, cn):
    results = []
    for i, g in enumerate(graphs):
        g = g.to(device)
        logits, (ei, alpha) = model(g.x, g.edge_index, g.edge_attr, return_attention=True)
        probs = torch.softmax(logits, dim=1); preds = probs.argmax(dim=1)
        pc = {}
        for p in preds.cpu().tolist():
            nm = cn[p] if p < len(cn) else f"class_{p}"; pc[nm] = pc.get(nm,0)+1
        if alpha.dim()>1: alpha=alpha.mean(1)
        k=min(5,alpha.shape[0]); tv,ti=torch.topk(alpha,k)
        te=[{"edge":(ei[0,j].item(),ei[1,j].item()),"attn":round(v,4)} for v,j in zip(tv.tolist(),ti.tolist())]
        mp = probs.max(1).values
        results.append({"graph":i,"nodes":g.x.shape[0],"preds":pc,"attack":any(p>0 for p in preds.cpu().tolist()),
                        "conf":round(float(mp.mean()),4),"top_attn":te})
    return results

def main():
    ap = argparse.ArgumentParser(description="Bước 3: Predict NHANH (không retrain)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", default="radar", choices=["uos","radar"])
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--model", default="EdgeGAT",
                    help="Tên model đã train bởi train_all.py (mặc định: EdgeGAT)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = get_device("cpu")
    # Hỗ trợ cả 2 layout: checkpoints/<model>/ (train_all.py) và checkpoints/ (train.py cũ)
    ckpt = os.path.join(args.ckpt_dir, args.model)
    if not os.path.exists(os.path.join(ckpt, "model_config.json")):
        ckpt = args.ckpt_dir
    t0=time.time(); model,mc=_load_model(ckpt,device); scaler=_load_scaler(args.data_dir)
    cn=mc["class_names"]; log.info("Model tải: %.2fs", time.time()-t0)
    t1=time.time(); gs=_convert(args.input,args.dataset,args.window)
    if not gs: log.error("0 graph từ %s",args.input); return
    gs=scaler.transform(gs); log.info("Convert: %d graph trong %.2fs",len(gs),time.time()-t1)
    t2=time.time(); res=predict(model,gs,device,cn); tp=time.time()-t2
    na=sum(1 for r in res if r["attack"])
    log.info("═══ KẾT QUẢ ═══")
    log.info("  File: %s | %d graph | %d có attack", args.input, len(res), na)
    log.info("  Thời gian: convert=%.2fs + predict=%.3fs = %.2fs TỔNG", time.time()-t1-tp, tp, time.time()-t1)
    for r in res[:3]:
        log.info("  Graph #%d: %d node, %s, conf=%.3f", r["graph"], r["nodes"], r["preds"], r["conf"])
    if args.out:
        with open(args.out,"w") as f: json.dump({"file":args.input,"graphs":len(res),"attack_graphs":na,"details":res},f,indent=2)
        log.info("  -> %s", args.out)

if __name__ == "__main__": main()
