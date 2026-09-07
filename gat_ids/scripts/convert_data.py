"""BƯỚC 1 — Chuyển CSV packet-level -> graph VÀ LƯU RA ĐĨA (chạy 1 lần).
Chạy: python scripts/convert_data.py --dataset radar --data-dir data/radar --window 5
"""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.common import load_config, set_seed, get_logger
from src.data.loader import load_from_csv, split_graphs, GraphScaler, DATASET_CLASSES
log = get_logger()

def main():
    ap = argparse.ArgumentParser(description="Bước 1: Convert CSV -> Graph + Lưu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split-by-file", default=None,
                    help="Chia tập theo TỆP thay vì ngẫu nhiên. Dạng: "
                         "'train:val:test' với danh sách tên tệp cách nhau bởi dấu phẩy. "
                         "Ví dụ: --split-by-file 0,1,2,3,4:5,6:7,8,9")
    ap.add_argument("--split-key", default="src_file",
                    choices=["src_file", "scenario"],
                    help="Với dataset=uos: chia theo tệp mô phỏng (src_file, "
                         "nghiêm ngặt nhất) hay theo nhóm kịch bản (scenario)")
    ap.add_argument("--skip-head", type=int, default=None,
                    help="Bỏ n dòng đầu mỗi tệp IoT-RPL (mặc định 300000, "
                         "vì nhiều tệp chia sẻ pha khởi động giống hệt nhau)")
    ap.add_argument("--split", default="random", choices=["random", "chrono"],
                    help="random = chia ngẫu nhiên phân tầng theo đồ thị (mặc "
                         "định cũ, CÓ rò rỉ thời gian); chrono = chia theo THỨ "
                         "TỰ THỜI GIAN trong từng tệp theo tỉ lệ 7:1:2, bảo đảm "
                         "max τ(train) < min τ(val) < min τ(test).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="tỉ lệ giữ lại của khung retained-ratio: giữ tiền tố "
                         "alpha đầu tiên theo thời gian của mỗi tệp trước khi "
                         "chia 7:1:2. Quét {0.5,0.6,0.8,1.0} để tách ảnh hưởng "
                         "của ngân sách dữ liệu khỏi dịch chuyển thời gian.")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    if args.skip_head is not None:
        os.environ["IOTRPL_SKIP_HEAD"] = str(args.skip_head)
    cfg = load_config(args.config)
    if args.dataset: cfg["data"]["dataset"] = args.dataset
    if args.data_dir: cfg["data"]["csv_dir"] = args.data_dir
    if args.window: cfg["data"]["window_seconds"] = args.window
    set_seed(cfg["seed"])
    ds, csv_dir, ws = cfg["data"]["dataset"], cfg["data"]["csv_dir"], cfg["data"]["window_seconds"]
    log.info("Dataset=%s | csv_dir=%s | window=%.1fs", ds, csv_dir, ws)
    graphs = load_from_csv(csv_dir, ds, ws)
    if not graphs and (ds == "synthetic" or cfg["data"].get("source") == "synthetic"):
        # SỬA LỖI: main.py có đường synthetic (get_datasets fallback) nhưng
        # pipeline 3 bước thì không -> không cách nào chạy thử convert/train_all/
        # test_all khi chưa có CSV thật. Dùng chung generate_dataset để hai
        # đường vào hành xử giống nhau.
        from src.data.synthetic import generate_dataset, CLASS_NAMES
        sc = cfg["data"]["synthetic"]
        log.info("Sinh synthetic: %d graph", sc["n_graphs"])
        graphs = generate_dataset(n_graphs=sc["n_graphs"],
                                  nodes_range=tuple(sc["nodes_per_graph"]),
                                  seed=cfg["seed"])
        cfg["data"]["classes"] = list(CLASS_NAMES)
    if not graphs: log.error("Không tạo được graph."); return
    class_names = DATASET_CLASSES.get(ds, cfg["data"]["classes"])   # ĐN trước split
    if args.split_by_file:
        parts = args.split_by_file.split(":")
        if len(parts) != 3:
            raise SystemExit("--split-by-file cần dạng 'train:val:test', "
                             "ví dụ 0,1,2,3,4:5,6:7,8,9")
        if ds == "uos":
            from src.data.uos_builder import split_uos_by_scenario
            train, val, test = split_uos_by_scenario(
                graphs, parts[0], parts[1], parts[2], key=args.split_key)
        else:
            from src.data.iotrpl_builder import split_by_file
            train, val, test = split_by_file(graphs, parts[0], parts[1], parts[2])
    elif args.split == "chrono":
        from src.utils.ata import chronological_split, temporal_leakage_gap
        train, val, test = chronological_split(graphs, alpha=args.alpha)
    else:
        train, val, test = split_graphs(graphs, tuple(cfg["data"]["split"]), cfg["seed"],
                                        num_classes=len(class_names))
    log.info("Chia: train=%d val=%d test=%d", len(train), len(val), len(test))
    # Chẩn đoán rò rỉ thời gian — in ra với MỌI chế độ chia để so sánh được.
    try:
        from src.utils.ata import temporal_leakage_gap as _tlg
        _g = _tlg(train, test)
        if _g != _g:      # NaN
            log.info("Chẩn đoán rò rỉ thời gian: không áp dụng được "
                     "(đồ thị không mang chỉ số cửa sổ).")
        else:
            log.info("Chẩn đoán rò rỉ thời gian: δ̄ = %.3f cửa sổ "
                     "(gần 0 => mẫu test nằm liền kề mẫu train theo thời gian)", _g)
        if args.split != "chrono" and _g == _g and _g < 2.0:
            log.warning("δ̄ nhỏ với phép chia NGẪU NHIÊN: điểm số thu được phản "
                        "ánh nhận dạng mẫu lặp lại hơn là khả năng bền vững "
                        "trước lưu lượng tương lai. Cân nhắc --split chrono.")
    except Exception as _e:
        log.debug("Bỏ qua chẩn đoán rò rỉ: %s", _e)
    # Yeo-Johnson (ổn định phương sai). Tắt: data.use_power=false
    if cfg["data"].get("use_power", True):
        from src.data.transforms import GraphPowerScaler
        scaler = GraphPowerScaler(use_power=True).fit(train)
    else:
        scaler = GraphScaler().fit(train)
    train = scaler.transform(train); val = scaler.transform(val); test = scaler.transform(test)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train, os.path.join(args.out_dir, "train.pt"))
    torch.save(val, os.path.join(args.out_dir, "val.pt"))
    torch.save(test, os.path.join(args.out_dir, "test.pt"))
    st = scaler.state_dict() if hasattr(scaler, "state_dict") else \
        {"x_mean":scaler.x_mean,"x_std":scaler.x_std,"e_mean":scaler.e_mean,"e_std":scaler.e_std}
    torch.save(st, os.path.join(args.out_dir, "scaler.pt"))
    actual = set(); [actual.update(g.y.unique().tolist()) for g in graphs]
    meta = {"dataset":ds,"window_seconds":ws,"num_graphs":{"train":len(train),"val":len(val),"test":len(test)},
            "n_node_features":int(train[0].x.shape[1]),"n_edge_features":int(train[0].edge_attr.shape[1]),
            "num_classes":len(class_names),"class_names":class_names,"classes_present":sorted(actual),
            "seed":cfg["seed"],"use_power":bool(cfg["data"].get("use_power",True))}
    # Kiểm toán rò rỉ nhãn (phải BÁO SẠCH với bộ đặc trưng mới)
    try:
        from src.data.feature_audit import (audit_leakage, RADAR_NODE_FEATURES_CLEAN21,
                                            RADAR_NODE_FEATURES_V29)
        d_n = int(train[0].x.shape[1])
        names = None
        if ds=="radar":
            if d_n==21: names = RADAR_NODE_FEATURES_CLEAN21
            elif d_n==29: names = RADAR_NODE_FEATURES_V29
        _, flagged = audit_leakage(train, names)
        meta["leakage_flagged"] = flagged
    except Exception as e:
        log.warning("Bỏ qua kiểm toán rò rỉ: %s", e)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("═══ ĐÃ LƯU ═══ train=%d val=%d test=%d -> %s", len(train), len(val), len(test), args.out_dir)
    log.info("Bước tiếp: python scripts/train.py")

if __name__ == "__main__": main()
