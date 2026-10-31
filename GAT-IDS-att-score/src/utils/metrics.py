"""Độ đo đánh giá: accuracy, per-class P/R/F1, macro-F1, confusion matrix.

Tự cài bằng numpy để không phụ thuộc bắt buộc vào sklearn (vẫn dùng được nếu có).
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Chạy model trên toàn loader, trả về (y_true, y_pred) dạng numpy."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = logits.argmax(dim=1)
        ys.append(batch.y.cpu())
        ps.append(pred.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return y_true, y_pred


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """Trả về precision, recall, f1 cho từng lớp từ confusion matrix."""
    num = cm.shape[0]
    prec = np.zeros(num)
    rec = np.zeros(num)
    f1 = np.zeros(num)
    for c in range(num):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                 if (prec[c] + rec[c]) > 0 else 0.0)
    return prec, rec, f1


def compute_metrics(y_true, y_pred, num_classes, class_names=None):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prec, rec, f1 = per_class_prf(cm)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    # Macro-F1: chỉ tính trên lớp CÓ MẪU trong y_true (tránh lệch khi thiếu lớp)
    present = [i for i in range(num_classes) if cm[i, :].sum() > 0]
    macro_f1 = float(np.mean([f1[i] for i in present])) if present else 0.0
    out = {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "per_class_f1": {(class_names[i] if class_names else i): float(f1[i])
                         for i in range(num_classes)},
        "per_class_precision": {(class_names[i] if class_names else i): float(prec[i])
                                for i in range(num_classes)},
        "per_class_recall": {(class_names[i] if class_names else i): float(rec[i])
                             for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
    }
    return out


def format_report(metrics, class_names):
    """In báo cáo dạng bảng gọn.

    SỬA LỖI: bản cũ bỏ qua mọi lớp có P=R=F1=0 với lý do "lớp không có mẫu".
    Nhưng một lớp CÓ MẪU mà model bỏ sót HOÀN TOÀN cũng có P=R=F1=0 — và bị
    giấu khỏi báo cáo. Đó đúng là trường hợp cần nhìn thấy nhất (lớp thiểu số
    không được phát hiện lần nào), lại là trường hợp bị ẩn đi.

    macro_f1 trong compute_metrics vẫn tính đúng (có cộng các số 0 đó), nên
    lỗi này chỉ ở phần HIỂN THỊ — nhưng nó khiến bảng in ra mâu thuẫn với
    macro-F1 và dễ dẫn tới kết luận sai.

    Bản mới quyết định theo SỐ MẪU THẬT lấy từ confusion matrix: lớp có mẫu
    thì luôn hiện (kèm cột support và dấu ! nếu bị bỏ sót hoàn toàn); chỉ lớp
    thực sự vắng mặt trong y_true mới bị ẩn.
    """
    import numpy as _np
    cm = _np.asarray(metrics.get("confusion_matrix", []))
    support = cm.sum(axis=1) if cm.size else None

    lines = []
    lines.append(f"  Accuracy : {metrics['accuracy']:.4f}")
    lines.append(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    lines.append("  " + "-" * 60)
    lines.append(f"  {'Class':<12}{'Prec':>8}{'Recall':>9}{'F1':>9}{'Support':>10}")
    missed = []
    for i, c in enumerate(class_names):
        p = metrics["per_class_precision"][c]
        r = metrics["per_class_recall"][c]
        f = metrics["per_class_f1"][c]
        n = int(support[i]) if support is not None and i < len(support) else None
        if n == 0:
            continue                      # lớp thực sự KHÔNG có mẫu trong test
        if support is None and p == 0 and r == 0 and f == 0:
            continue                      # không có cm -> giữ hành vi cũ
        flag = ""
        if n and f == 0.0:
            flag = "  <-- BỎ SÓT HOÀN TOÀN"
            missed.append(c)
        sup = f"{n:>10d}" if n is not None else " " * 10
        lines.append(f"  {c:<12}{p:>8.3f}{r:>9.3f}{f:>9.3f}{sup}{flag}")
    if missed:
        lines.append(f"  ! {len(missed)} lớp không được phát hiện lần nào: "
                     + ", ".join(missed))
    return "\n".join(lines)
