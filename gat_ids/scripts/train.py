"""BƯỚC 2 — Train model từ graph ĐÃ CONVERT (chạy 1 lần).
Chạy: python scripts/train.py [--federated] [--epochs 100] [--no-edge]
"""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.loader import make_loaders
from src.models.edge_gat import build_model
from src.utils.engine import make_criterion, train_centralized
from src.federated.fed import train_federated
log = get_logger()

def main():
    ap = argparse.ArgumentParser(description="Bước 2: Train từ graph đã convert")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--federated", action="store_true")
    ap.add_argument("--no-edge", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.no_edge: cfg["model"]["use_edge_features"] = False
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.federated: cfg["federated"]["enabled"] = True
    cfg["train"]["ckpt_dir"] = args.ckpt_dir
    set_seed(cfg["seed"]); device = get_device(cfg["device"])
    tp = os.path.join(args.data_dir, "train.pt")
    if not os.path.exists(tp): log.error("Chưa convert. Chạy: python scripts/convert_data.py"); return
    train_g = torch.load(tp, weights_only=False)
    val_g = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=False)
    test_g = torch.load(os.path.join(args.data_dir, "test.pt"), weights_only=False)
    with open(os.path.join(args.data_dir, "meta.json"), "r") as f: mr = json.load(f)
    meta = {"num_classes":mr["num_classes"],"class_names":mr["class_names"],"n_node_features":mr["n_node_features"],"n_edge_features":mr["n_edge_features"]}
    log.info("Tải: train=%d val=%d test=%d | d_n=%d d_e=%d", len(train_g), len(val_g), len(test_g), meta["n_node_features"], meta["n_edge_features"])
    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"])
    model = build_model(cfg, meta).to(device)
    log.info("Model: %d params", model.count_parameters())
    criterion = make_criterion(cfg, train_g, meta["num_classes"], device)
    if cfg["federated"]["enabled"]:
        model, test_m = train_federated(model, train_g, loaders[1], loaders[2], criterion, cfg, meta, device)
    else:
        model, test_m = train_centralized(model, loaders, criterion, cfg, meta, device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pt"))
    mcfg = {"in_dim":meta["n_node_features"],"edge_dim":meta["n_edge_features"],"hidden_dim":cfg["model"]["hidden_dim"],
            "out_dim":cfg["model"]["out_dim"],"num_classes":meta["num_classes"],"heads":cfg["model"]["heads"],
            "dropout":cfg["model"]["dropout"],"use_edge_features":cfg["model"]["use_edge_features"],
            "use_dyt":cfg["model"].get("use_dyt",True),"class_names":meta["class_names"]}
    with open(os.path.join(args.ckpt_dir, "model_config.json"), "w") as f: json.dump(mcfg, f, indent=2)
    log.info("═══ ĐÃ LƯU MODEL -> %s ═══", args.ckpt_dir)
    log.info("Bước tiếp: python scripts/predict.py --input mau_moi.csv")

if __name__ == "__main__": main()
