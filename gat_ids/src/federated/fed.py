"""Federated Learning: phân vùng client (IID / non-IID Dirichlet) + FedAvg/FedProx.

Cài đặt gọn, tự chứa, để hiểu rõ cơ chế. Có thể thay bằng Flower sau mà
không đổi phần model/data. Mỗi client = 1 edge gateway giữ một phần graph.
"""
from __future__ import annotations
import copy
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from ..utils.metrics import compute_metrics, collect_predictions, format_report
from ..utils.engine import train_one_epoch
from ..utils.common import get_logger

log = get_logger()


# ──────────────────────────────────────────────────────────────
#  Phân vùng dữ liệu cho các client
# ──────────────────────────────────────────────────────────────
def _graph_label(g):
    vals, counts = torch.unique(g.y, return_counts=True)
    return int(vals[counts.argmax()])


def partition_clients(train_graphs, num_clients, mode="dirichlet",
                      alpha=0.5, num_classes=6, seed=42):
    """Chia train_graphs cho num_clients.

    - iid: chia ngẫu nhiên đều.
    - dirichlet: mô phỏng non-IID — mỗi client lệch về một số lớp,
      alpha nhỏ => lệch mạnh (thực tế hơn nhưng khó hội tụ hơn).
    """
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(num_clients)]

    if mode == "iid":
        idx = rng.permutation(len(train_graphs))
        for j, i in enumerate(idx):
            clients[j % num_clients].append(train_graphs[i])
        return clients

    # Dirichlet theo nhãn graph
    by_label = {}
    for g in train_graphs:
        by_label.setdefault(_graph_label(g), []).append(g)

    for label, items in by_label.items():
        rng.shuffle(items)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(items)).astype(int)[:-1]
        start = 0
        for cid in range(num_clients):
            end = cuts[cid] if cid < len(cuts) else len(items)
            clients[cid].extend(items[start:end])
            start = end

    for cid in range(num_clients):
        sizes = len(clients[cid])
        log.info("  Client %d: %d graph", cid, sizes)
    return clients


# ──────────────────────────────────────────────────────────────
#  Aggregation
# ──────────────────────────────────────────────────────────────
def fedavg(global_model, client_states, client_sizes):
    """Trung bình hóa có trọng số theo số mẫu (FedAvg)."""
    total = float(sum(client_sizes))
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state.keys():
        if new_state[key].dtype.is_floating_point:
            agg = torch.zeros_like(new_state[key], dtype=torch.float32)
            for st, sz in zip(client_states, client_sizes):
                agg += st[key].float() * (sz / total)
            new_state[key] = agg.to(new_state[key].dtype)
        else:
            # tham số không phải float (hiếm) -> lấy của client lớn nhất
            new_state[key] = client_states[int(np.argmax(client_sizes))][key]
    global_model.load_state_dict(new_state)
    return global_model


# ──────────────────────────────────────────────────────────────
#  Vòng huấn luyện Federated
# ──────────────────────────────────────────────────────────────
def train_federated(global_model, train_graphs, val_loader, test_loader,
                    criterion, cfg, meta, device):
    fcfg = cfg["federated"]
    tcfg = cfg["train"]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    clients = partition_clients(
        train_graphs, fcfg["num_clients"], fcfg["partition"],
        fcfg["dirichlet_alpha"], num_classes, cfg["seed"])
    client_sizes = [len(c) for c in clients]

    best_val = -1.0
    best_state = copy.deepcopy(global_model.state_dict())
    use_fedprox = fcfg["aggregator"] == "fedprox"
    mu = fcfg["fedprox_mu"] if use_fedprox else 0.0

    for rnd in range(1, fcfg["rounds"] + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        client_states = []

        for cid, graphs in enumerate(clients):
            if len(graphs) == 0:
                client_states.append(copy.deepcopy(global_state))
                continue
            # Khởi tạo client từ global
            local = copy.deepcopy(global_model).to(device)
            opt = torch.optim.Adam(local.parameters(), lr=tcfg["lr"],
                                   weight_decay=tcfg["weight_decay"])
            loader = DataLoader(graphs, batch_size=tcfg["batch_size"], shuffle=True)
            for _ in range(fcfg["local_epochs"]):
                train_one_epoch(local, loader, opt, criterion, device,
                                fedprox_mu=mu, global_state=global_state)
            client_states.append({k: v.cpu() for k, v in local.state_dict().items()})

        # Server tổng hợp
        global_model = fedavg(global_model, client_states, client_sizes)

        # Đánh giá trên val
        val_m = compute_metrics(*collect_predictions(global_model, val_loader, device),
                                num_classes, class_names)
        if val_m["macro_f1"] > best_val:
            best_val = val_m["macro_f1"]
            best_state = copy.deepcopy(global_model.state_dict())
        # SỬA: chỉ log mỗi 5 vòng -> vòng CUỐI thường không được in,
        # người dùng không thấy trạng thái hội tụ khi rounds không chia hết cho 5.
        if rnd % 5 == 0 or rnd == 1 or rnd == fcfg["rounds"]:
            log.info("Round %3d | val macroF1 %.4f (best %.4f)",
                     rnd, val_m["macro_f1"], best_val)

    # Test với global model tốt nhất
    global_model.load_state_dict(best_state)
    test_m = compute_metrics(*collect_predictions(global_model, test_loader, device),
                             num_classes, class_names)
    log.info("== KẾT QUẢ TEST (federated, %s, %s) ==\n%s",
             fcfg["aggregator"], fcfg["partition"],
             format_report(test_m, class_names))
    return global_model, test_m
