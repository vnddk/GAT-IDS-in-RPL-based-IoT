"""BƯỚC 2 — Train TẤT CẢ models từ graph đã convert, lưu từng checkpoint.

Chạy: python scripts/train_all.py [--data-dir data/processed] [--epochs 60]

Tải graph từ data/processed/ (output Bước 1), train lần lượt 6 model,
lưu mỗi model vào checkpoints/<model_name>/best_model.pt

Sau bước này:
  checkpoints/
    RF/model.pkl
    XGBoost/model.pkl
    MLP/best_model.pt + model_config.json
    GCN/best_model.pt + model_config.json
    E-GraphSAGE/best_model.pt + model_config.json
    EdgeGAT/best_model.pt + model_config.json
"""
import os, sys, json, argparse, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.common import load_config, set_seed, get_device, get_logger
from src.data.loader import make_loaders
from src.models.edge_gat import build_model as build_gat
from src.models.gnn_baselines import build_baseline
from src.models.tabular_baselines import _flatten_graphs
from src.utils.engine import make_criterion, train_centralized

log = get_logger()


def _load_data(data_dir):
    train = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
    val = torch.load(os.path.join(data_dir, "val.pt"), weights_only=False)
    test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)
    with open(os.path.join(data_dir, "meta.json")) as f:
        mr = json.load(f)
    meta = {"num_classes": mr["num_classes"], "class_names": mr["class_names"],
            "n_node_features": mr["n_node_features"], "n_edge_features": mr["n_edge_features"]}
    return train, val, test, meta


def _build_negat(cfg, meta):
    """Dựng NE-GAT-IDS, đọc các cờ ablation từ cfg["model"]."""
    from src.models.ne_gat import build_negat
    return build_negat(cfg, meta)


def _save_gnn(model, cfg, meta, ckpt_dir, name, seed=42):
    d = os.path.join(ckpt_dir, name); os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "best_model.pt"))
    mc = {"in_dim": meta["n_node_features"], "edge_dim": meta["n_edge_features"],
          "hidden_dim": cfg["model"]["hidden_dim"], "out_dim": cfg["model"]["out_dim"],
          "num_classes": meta["num_classes"], "heads": cfg["model"]["heads"],
          "dropout": cfg["model"]["dropout"], "use_edge_features": cfg["model"]["use_edge_features"],
          "use_dyt": cfg["model"].get("use_dyt", True),
          "use_input_skip": cfg["model"].get("use_input_skip", True),
          "skip_mode": cfg["model"].get("skip_mode", "raw"),
          "num_layers": cfg["model"].get("num_layers", 2),
          "layer_residual": cfg["model"].get("layer_residual", False),
          "use_main_residual": cfg["model"].get("use_main_residual", True),
          "classifier_type": cfg["model"].get("classifier_type", "linear"),
          "clf_hidden": cfg["model"].get("clf_hidden", 128),
          "edge_latent": cfg["model"].get("edge_latent", 24),
          "node_heads": cfg["model"].get("node_heads", 2),
          "edge_score_mode": cfg["model"].get("edge_score_mode", "full"),
          "edge_agg": cfg["model"].get("edge_agg", "mean_deg"),
          "edge_update": cfg["model"].get("edge_update", True),
          "seed": seed,
          "class_names": meta["class_names"],
          "model_type": name}
    with open(os.path.join(d, "model_config.json"), "w") as f:
        json.dump(mc, f, indent=2)
    log.info("  Saved -> %s", d)


def main():
    ap = argparse.ArgumentParser(description="Bước 2: Train tất cả models")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    # ── NE-GAT-IDS (kiến trúc hai kênh) ──
    ap.add_argument("--edge-score", default="full", choices=["full", "edge_only"],
                    help="NE-GAT: cách tính điểm chú ý của kênh cạnh. "
                         "full = dùng [h_i||h_j||e_ij] như tài liệu gốc; "
                         "edge_only = CHỈ dùng e_ij (đã sửa, để thật sự cô lập "
                         "kênh cạnh khỏi biểu diễn nút — xem SỬA LỖI 2).")
    ap.add_argument("--edge-agg", default="mean_deg", choices=["mean_deg", "mean"],
                    help="NE-GAT: cách tổng hợp kênh cạnh. mean = trung bình có "
                         "trọng số như tài liệu gốc (MẤT thông tin độ lớn); "
                         "mean_deg = bổ sung thành phần theo bậc (mặc định, SỬA LỖI 1).")
    ap.add_argument("--no-edge-update", action="store_true",
                    help="NE-GAT: TẮT cập nhật trạng thái cạnh qua từng tầng "
                         "(ablation D so với C trong tài liệu thiết kế).")
    ap.add_argument("--edge-latent", type=int, default=24,
                    help="NE-GAT: số chiều tiềm ẩn của cạnh (mặc định 24)")
    ap.add_argument("--node-heads", type=int, default=2,
                    help="NE-GAT: số đầu chú ý của kênh nút (mặc định 2)")
    ap.add_argument("--no-edge", action="store_true",
                    help="ABLATION: tắt đặc trưng cạnh trong cơ chế chú ý. "
                         "Kiến trúc, số tầng, hàm mất mát, hạt giống giữ NGUYÊN "
                         "— chỉ véc-tơ cạnh không còn tham gia công thức tính "
                         "hệ số chú ý. Dùng cặp lệnh có/không cờ này để đo đóng "
                         "góp riêng của đặc trưng cạnh.")
    ap.add_argument("--lgbm-weight", default="cb", choices=["cb", "balanced", "none"],
                    help="sơ đồ trọng số lớp cho LightGBM. cb = cân bằng theo số "
                         "mẫu hiệu dụng có chặn trần (mặc định, khuyến nghị); "
                         "balanced = nghịch tần đầy đủ (ĐÃ BIẾT gây sụp trên RADAR); "
                         "none = không trọng số.")
    ap.add_argument("--lgbm-weight-cap", type=float, default=3.0,
                    help="trần tỉ số trọng số khi --lgbm-weight cb (mặc định 3,0)")
    ap.add_argument("--epochs", type=int, default=60)
    # MẶC ĐỊNH ĐỔI smote -> balance (v3.26).
    # Lý do: phương pháp cân bằng lớp của luận văn là NHÂN BẢN đồ thị đã có,
    # KHÔNG sinh dữ liệu giả. Mặc định cũ là "smote", vốn nội suy ra véc-tơ
    # đặc trưng chưa từng tồn tại (src/data/loader.py, x_syn = X[i] + lam*(X[j]-X[i])),
    # nên mọi lần chạy trước đây đã âm thầm dùng SMOTE dù không ai gõ cờ đó.
    ap.add_argument("--resample", default="balance",
                    choices=["balance", "ratio", "smote", "weighted", "off"],
                    help="balance: nhân bản đồ thị đưa mọi lớp tấn công về mức "
                         "lớp tấn công lớn nhất (MẶC ĐỊNH — không sinh dữ liệu giả); "
                         "ratio: nhân bản theo ngưỡng (legacy v1, đã làm rank F1 "
                         "sụp 0,676->0,199); smote: nội suy sinh node tổng hợp; "
                         "weighted: lấy mẫu có trọng số; off: tắt")
    ap.add_argument("--skip-mode", default=None,
                    choices=["none","raw","hybrid"],
                    help="Ghi đè model.skip_mode trong config (phục vụ ablation)")
    ap.add_argument("--models", default="all",
                    help="Model cần train, cách nhau bởi dấu phẩy. "
                         "Ví dụ: --models EdgeGAT (chỉ mô hình đề xuất). "
                         "Tên hợp lệ: RF,XGBoost,LightGBM,MLP,GCN,E-GraphSAGE,EdgeGAT,NE-GAT,all")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Số tầng GAT của EdgeGAT (2 mặc định, 3 = thử 3-hop)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Hạt giống ngẫu nhiên. Chạy nhiều seed rồi báo cáo "
                         "trung bình ± độ lệch chuẩn.")
    ap.add_argument("--classifier", default=None,
                    choices=["linear", "mlp", "lstm"],
                    help="Kiểu tầng phân loại cuối của EdgeGAT")
    ap.add_argument("--clf-hidden", type=int, default=None,
                    help="Chiều ẩn cho classifier mlp/lstm (mặc định 64)")
    ap.add_argument("--no-main-residual", action="store_true",
                    help="TẮT phần dư chính (h_L + W_res·h0). Dùng cho ablation; "
                         "KHÔNG ảnh hưởng kết nối tắt đặc trưng gốc.")
    ap.add_argument("--layer-residual", action="store_true",
                    help="Bật phần dư từng tầng (chống làm mịn quá mức khi ≥3 tầng)")
    ap.add_argument("--weight-cap", type=float, default=None,
                    help="Ghi đè train.weight_cap: trần tỉ lệ trọng số lớp "
                         "max/min (3|5|10, 0=tắt). Chống bù quá đà.")
    ap.add_argument("--cb-beta", type=float, default=None,
                    help="Ghi đè train.cb_beta. Nhỏ hơn = hiệu chỉnh NHẸ hơn = "
                         "accuracy cao hơn, macro-F1 có thể thấp hơn. "
                         "0.9999 (mặc định, tỉ lệ 18x) | 0.999 (2.3x) | 0.99 (1x)")
    ap.add_argument("--sampler-cap", type=float, default=5.0,
                    help="Trần tỉ lệ trọng số max/min của resample=weighted "
                         "(5=mặc định, 0=tắt trần). Chống lớp hiếm vọt quá xa.")
    ap.add_argument("--sampler-alpha", type=float, default=0.5,
                    help="Độ mạnh của resample=weighted (0=tắt, 0.5=ôn hoà, 1=cân bằng hẳn)")
    ap.add_argument("--oversample", type=float, default=0.02,
                    help="Ngưỡng cho resample=ratio (mặc định 0.02 = 2%%)")
    args = ap.parse_args()
    # Chọn model cần train (giữ NGUYÊN SMOTE + bố cục checkpoint để so sánh
    # được với các lần chạy đầy đủ và để test_all.py đọc đúng thư mục con).
    _ALL = ["RF", "XGBoost", "LightGBM", "MLP", "GCN", "E-GraphSAGE", "EdgeGAT", "NE-GAT"]
    _alias = {m.lower().replace("-", ""): m for m in _ALL}
    if args.models.strip().lower() in ("all", ""):
        WANT = set(_ALL)
    else:
        WANT = set()
        for tok in args.models.split(","):
            k = tok.strip().lower().replace("-", "")
            if k in _alias:
                WANT.add(_alias[k])
            elif k:
                raise SystemExit("Không có model '%s'. Hợp lệ: %s"
                                 % (tok.strip(), ",".join(_ALL)))

    cfg = load_config(args.config)

    cfg["model"]["edge_score_mode"] = args.edge_score

    cfg["model"]["edge_agg"] = args.edge_agg

    cfg["model"]["edge_update"] = not args.no_edge_update

    cfg["model"]["edge_latent"] = args.edge_latent

    cfg["model"]["node_heads"] = args.node_heads

    if getattr(args, "no_edge", False):

        cfg["model"]["use_edge_features"] = False

        log.warning("ABLATION --no-edge: TẮT đặc trưng cạnh trong hệ số chú ý. "

                    "Mọi thành phần khác giữ nguyên.")
    if args.num_layers is not None:
        cfg["model"]["num_layers"] = args.num_layers
        log.info("Ghi đè num_layers = %d (%d-hop)", args.num_layers, args.num_layers)
    if args.seed is not None:
        cfg["seed"] = args.seed
        log.info("Hạt giống = %d", args.seed)
    SEED = cfg.get("seed", 42)
    if args.classifier is not None:
        cfg["model"]["classifier_type"] = args.classifier
        log.info("Classifier = %s", args.classifier)
    if args.clf_hidden is not None:
        cfg["model"]["clf_hidden"] = args.clf_hidden
    if args.no_main_residual:
        cfg["model"]["use_main_residual"] = False
        log.info("TẮT phần dư chính (ablation)")
    if args.layer_residual:
        cfg["model"]["layer_residual"] = True
        log.info("Bật phần dư từng tầng")
    if args.weight_cap is not None:
        cfg["train"]["weight_cap"] = args.weight_cap
        log.info("Ghi đè weight_cap = %s", args.weight_cap)
    if args.cb_beta is not None:
        cfg["train"]["cb_beta"] = args.cb_beta
        log.info("Ghi đè cb_beta = %s", args.cb_beta)
    if args.skip_mode is not None:
        cfg["model"]["skip_mode"] = args.skip_mode
        cfg["model"]["use_input_skip"] = (args.skip_mode != "none")
        log.info("Ghi đè skip_mode = %s", args.skip_mode)
    cfg["train"]["epochs"] = args.epochs
    device = get_device(cfg["device"])

    if not os.path.exists(os.path.join(args.data_dir, "train.pt")):
        log.error("Chưa convert. Chạy Bước 1: python scripts/convert_data.py"); return
    train_g, val_g, test_g, meta = _load_data(args.data_dir)
    nc, cn = meta["num_classes"], meta["class_names"]
    log.info("Tải: train=%d val=%d test=%d | d_n=%d d_e=%d",
             len(train_g), len(val_g), len(test_g), meta["n_node_features"], meta["n_edge_features"])

    # Xử lý mất cân bằng lớp (áp dụng cho CẢ 6 model để so sánh công bằng)
    from src.utils.engine import _node_counts
    orig_counts = _node_counts(train_g, nc)   # phân bố GỐC, trước mọi resample
    sampler = None
    if args.resample == "weighted":
        from src.data.loader import build_weighted_sampler
        sampler = build_weighted_sampler(train_g, nc, alpha=args.sampler_alpha,
                                         seed=cfg.get("seed", 42),
                                         cap=args.sampler_cap)
    elif args.resample == "smote":
        from src.data.loader import smote_augment_graphs
        set_seed(SEED)
        log.warning("--resample smote: SINH DỮ LIỆU GIẢ bằng nội suy đặc trưng. "
                    "Đây KHÔNG phải phương pháp cân bằng lớp của luận văn "
                    "(dùng --resample balance). Chỉ dùng để so sánh/ablation.")
        train_g = smote_augment_graphs(train_g, nc)
    elif args.resample in ("balance", "ratio"):
        from src.data.loader import oversample_rare_classes
        set_seed(SEED)
        train_g = oversample_rare_classes(train_g, nc, min_ratio=args.oversample,
                                           mode=args.resample)

    else:
        log.info("Resample: TẮT (dùng phân bố lớp gốc)")
    log.info("Cân bằng lớp: --resample %s | phân bố sau xử lý: %s",
             args.resample,
             {int(c): int(v) for c, v in enumerate(_node_counts(train_g, nc).tolist())})

    loaders = make_loaders(train_g, val_g, test_g, cfg["train"]["batch_size"], sampler=sampler)

    results = {}

    # ── 1. Random Forest ──────────────────────────────────────────
    X_tr = y_tr = None
    if {"RF", "XGBoost", "LightGBM"} & WANT:
        X_tr, y_tr = _flatten_graphs(train_g)   # dùng chung cho RF và XGBoost
    if "RF" in WANT:
      log.info("\n=== Random Forest ===")
      from sklearn.ensemble import RandomForestClassifier
      t0 = time.time()
      set_seed(SEED)
      clf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced",
                                    n_jobs=-1, random_state=42)
      clf.fit(X_tr, y_tr)
      d = os.path.join(args.ckpt_dir, "RF"); os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, "model.pkl"), "wb") as f: pickle.dump(clf, f)
      log.info("  RF trained in %.1fs -> %s", time.time()-t0, d)

    # ── 2. XGBoost ────────────────────────────────────────────────
    if "XGBoost" in WANT:
      log.info("\n=== XGBoost ===")
      from xgboost import XGBClassifier
      import numpy as np
      t0 = time.time()
      set_seed(SEED)
      present = sorted(set(y_tr.tolist()))
      remap = {c: i for i, c in enumerate(present)}
      y_tr_r = np.array([remap[c] for c in y_tr])
      clf2 = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                            eval_metric='mlogloss', tree_method='hist', random_state=42, n_jobs=-1)
      clf2.fit(X_tr, y_tr_r)
      d = os.path.join(args.ckpt_dir, "XGBoost"); os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, "model.pkl"), "wb") as f:
          pickle.dump({"model": clf2, "remap": remap}, f)
      log.info("  XGBoost trained in %.1fs -> %s", time.time()-t0, d)

    # ── 2b. LightGBM ──────────────────────────────────────────────
    if "LightGBM" in WANT:
      log.info("\n=== LightGBM ===")
      from lightgbm import LGBMClassifier
      import numpy as np
      t0 = time.time()
      set_seed(SEED)
      present3 = sorted(set(y_tr.tolist()))
      remap3 = {c: i for i, c in enumerate(present3)}
      y_tr_l = np.array([remap3[c] for c in y_tr])
      # class_weight='balanced' để tương đương cách xử lý mất cân bằng của RF;
      # XGBoost không có tham số này nên ba mô hình cây bổ trợ nhau về góc nhìn.
      # ══ TRỌNG SỐ LỚP CHO LIGHTGBM ══════════════════════════════
      # Bằng chứng từ scripts/debug_lightgbm.py trên bộ RADAR (16 lớp,
      # tỉ lệ mất cân bằng 529:1):
      #     class_weight='balanced'  -> acc(TRAIN)=0,0912  acc(TEST)=0,0922
      #     bỏ hẳn trọng số          -> acc(TEST)=0,8472  macro-F1=0,3235
      # Mô hình KHÔNG khớp nổi ngay cả tập huấn luyện, tức đây là hỏng ở
      # khâu HỌC. Thủ phạm là 'balanced': nó cho tỉ số trọng số lớn nhất
      # trên nhỏ nhất bằng đúng 529, khiến hessian trong LightGBM lệch tới
      # mức không tách được nút. RandomForest chịu được (0,8463) vì cây
      # đơn không dùng gradient/hessian.
      #
      # Bỏ hẳn trọng số thì học được nhưng macro-F1 chỉ 0,3235 — lớp thiểu
      # số bị bỏ rơi. Do đó dùng phương án ở giữa: trọng số CÂN BẰNG THEO
      # SỐ MẪU HIỆU DỤNG có CHẶN TRẦN, đúng công thức (2.7)-(2.8) mà
      # EdgeGAT đang dùng cho CB-Focal. Trần 3,0 giới hạn tỉ số trọng số
      # còn 3:1 thay vì 529:1.
      #
      # Lợi ích phụ về mặt phương pháp: cả EdgeGAT lẫn LightGBM khi đó
      # dùng CÙNG một sơ đồ trọng số lớp, nên bảng so sánh công bằng hơn.
      #
      # CHƯA XÁC ĐỊNH ĐƯỢC trần nào tốt nhất trên RADAR — hãy quét
      # {2, 3, 5, 8} bằng --lgbm-weight-cap rồi chốt theo macro-F1 val.
      import numpy as _np
      _mode = getattr(args, "lgbm_weight", "cb")
      _cap = float(getattr(args, "lgbm_weight_cap", 3.0))
      _cw, _sw = None, None
      if _mode == "balanced":
          _cw = "balanced"
          log.warning("  LightGBM: dùng class_weight='balanced' — đã biết gây "
                      "sụp trên RADAR. Chỉ dùng để tái lập lỗi.")
      elif _mode == "cb":
          from src.utils.losses import class_balanced_weights
          _cnt = _np.bincount(y_tr_l, minlength=len(present3))
          _w = class_balanced_weights(torch.tensor(_cnt, dtype=torch.double),
                                      beta=cfg["train"].get("cb_beta", 0.9999),
                                      cap=_cap).numpy()
          _sw = _w[y_tr_l]
          log.info("  LightGBM: trọng số CB có trần %.1f | w_min=%.3f w_max=%.3f "
                   "(tỉ số %.1f:1, so với %.0f:1 của 'balanced')",
                   _cap, _w.min(), _w.max(), _w.max()/max(_w.min(), 1e-9),
                   _cnt.max()/max(_cnt.min(), 1))
      else:
          log.info("  LightGBM: KHÔNG dùng trọng số lớp")

      # Bỏ `objective="multiclass"` và `num_class=` truyền tay:
      # LGBMClassifier tự suy ra hai tham số này từ nhãn; truyền thừa là một
      # nguồn lỗi đã biết của lớp bọc sklearn. (Phép thử [3] xác nhận việc bỏ
      # chúng KHÔNG tự nó sửa được lỗi — nguyên nhân là trọng số ở trên.)
      clf3 = LGBMClassifier(
          n_estimators=300, num_leaves=63, max_depth=-1, learning_rate=0.08,
          subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
          min_child_samples=10, class_weight=_cw,
          random_state=42, n_jobs=-1, verbose=-1)
      clf3.fit(X_tr, y_tr_l, sample_weight=_sw)
      # KIỂM TRA NGAY SAU KHI HỌC: độ chính xác trên chính TẬP HUẤN LUYỆN.
      # Không có bước này thì một mô hình hỏng chỉ lộ ra ở Bước 3, khi đã
      # quá muộn để biết hỏng ở khâu học hay khâu dự đoán. Ngưỡng 0,50 đặt
      # rất thấp: mọi mô hình cây lành mạnh đều vượt xa trên tập nó vừa học.
      _acc_tr = float((clf3.predict(X_tr) == y_tr_l).mean())
      log.info("  LightGBM: accuracy trên TẬP HUẤN LUYỆN = %.4f", _acc_tr)
      if _acc_tr < 0.50:
          log.warning("  !! LightGBM không khớp nổi tập huấn luyện (%.4f). "
                      "Kết quả kiểm thử sẽ vô nghĩa — chạy "
                      "scripts/debug_lightgbm.py để khoanh vùng nguyên nhân.",
                      _acc_tr)
      d = os.path.join(args.ckpt_dir, "LightGBM"); os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, "model.pkl"), "wb") as f:
          pickle.dump({"model": clf3, "remap": remap3}, f)
      log.info("  LightGBM trained in %.1fs -> %s", time.time()-t0, d)

    # ── 3-6. GNN models ──────────────────────────────────────────
    gnn_models = [
        ("MLP", lambda: build_baseline("mlp", meta, cfg)),
        ("GCN", lambda: build_baseline("gcn", meta, cfg)),
        ("E-GraphSAGE", lambda: build_baseline("edge_graphsage", meta, cfg)),
        ("EdgeGAT", lambda: build_gat(cfg, meta)),
        ("NE-GAT", lambda: _build_negat(cfg, meta)),
    ]
    gnn_models = [(n, f) for n, f in gnn_models if n in WANT]
    for i, (name, build_fn) in enumerate(gnn_models, 1):
        log.info("\n=== [%d/%d] %s ===", i, len(gnn_models), name)
        set_seed(SEED)
        model = build_fn().to(device)
        log.info("  %s: %d params", name, model.count_parameters())
        criterion = make_criterion(cfg, train_g, nc, device,
                                   orig_counts=orig_counts)
        model, _ = train_centralized(model, loaders, criterion, cfg, meta, device)
        _save_gnn(model, cfg, meta, args.ckpt_dir, name, seed=SEED)

    log.info("\n═══ BƯỚC 2 HOÀN TẤT ═══")
    log.info("Tất cả model đã lưu tại %s/", args.ckpt_dir)
    log.info("Bước tiếp: python scripts/test_all.py")


if __name__ == "__main__":
    main()
