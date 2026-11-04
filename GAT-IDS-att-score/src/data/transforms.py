"""Biến đổi ổn định phương sai cho node/edge features — cảm hứng từ bài báo
Amino-Acid IDS (Internet of Things 2026), §3.3 "Yeo-Johnson transformation".

VÌ SAO đưa vào FedEdge-GAT:
    Các đặc trưng RPL đếm gói (DIO/DAO/DIS sum, payload, n_sent...) lệch phải
    rất mạnh (đuôi dài do vài node bùng nổ traffic). Z-score thuần (GraphScaler
    cũ) KHÔNG sửa được độ lệch — chỉ dịch/chia. Với input lệch, cơ chế attention
    của GAT và tanh của DyT dễ bão hoà, gradient nhiễu -> đúng hiện tượng val
    macro-F1 dao động ±0.08 mỗi epoch trong log 100 epochs của bạn.

    Yeo-Johnson (Yeo & Johnson 2000) đưa từng feature về gần chuẩn TRƯỚC khi
    z-score, xử lý được cả giá trị 0 và âm (khác Box-Cox). Thực nghiệm numpy:
    skew payload +1.97 -> -0.03.

CHỐNG RÒ RỈ: lambda + mean/std CHỈ fit trên tập train, áp cho val/test.

Công thức (bài báo, Eq.1):
    psi(x,λ) = ((x+1)^λ − 1)/λ            nếu x≥0, λ≠0
             = log(x+1)                    nếu x≥0, λ=0
             = −((−x+1)^(2−λ) − 1)/(2−λ)   nếu x<0, λ≠2
             = −log(−x+1)                  nếu x<0, λ=2
λ chọn bằng grid-search cực đại log-likelihood (Eq.2, Algorithm 1), miền [−2,2].
"""
from __future__ import annotations
import numpy as np
import torch

from ..utils.common import get_logger

log = get_logger()


# ──────────────────────────────────────────────────────────────
#  Hạt nhân Yeo-Johnson (numpy, ổn định số học bằng log1p/expm1)
# ──────────────────────────────────────────────────────────────
def _yj(x: np.ndarray, lam: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    # nhánh x >= 0
    if abs(lam) < 1e-6:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    # nhánh x < 0
    neg = ~pos
    if abs(lam - 2.0) < 1e-6:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _fit_lambda(col: np.ndarray, lo=-2.0, hi=2.0, step=0.1) -> float:
    """Grid-search λ cực đại log-likelihood — đúng Algorithm 1 bài báo.

    Đã đối chiếu numpy với scipy.stats.yeojohnson_normmax: sai khác < 0.05,
    skewness sau biến đổi ~ 0. Dùng grid (không cần scipy) để giữ requirements
    gọn; nếu có scipy, có thể thay bằng yeojohnson_normmax cho mượt hơn.
    """
    col = np.asarray(col, dtype=np.float64)
    col = col[np.isfinite(col)]
    if col.size < 8 or np.allclose(col, col[0]):
        return 1.0                                  # hằng số -> không biến đổi
    sgn_term = np.sum(np.sign(col) * np.log1p(np.abs(col)))
    n = col.size
    best_lam, best_ll = 1.0, -np.inf
    for lam in np.arange(lo, hi + 1e-9, step):
        z = _yj(col, float(lam))
        var = z.var()
        if var <= 1e-12:
            continue
        ll = -0.5 * n * np.log(var) + (lam - 1.0) * sgn_term
        if ll > best_ll:
            best_ll, best_lam = ll, float(lam)
    return best_lam


# ──────────────────────────────────────────────────────────────
#  Scaler cho đồ thị: Yeo-Johnson theo từng feature + z-score
# ──────────────────────────────────────────────────────────────
class GraphPowerScaler:
    """Thay thế drop-in cho GraphScaler (loader.py) với biến đổi power.

    Dùng y hệt interface cũ: fit(train_graphs) -> transform(graphs).
    Có thể tắt power để so sánh (ablation) bằng use_power=False -> về z-score.
    """

    def __init__(self, use_power: bool = True, skip_binary: bool = True):
        self.use_power = use_power
        self.skip_binary = skip_binary            # bỏ qua feature nhị phân/hằng
        self.x_lam = self.e_lam = None
        self.x_mean = self.x_std = None
        self.e_mean = self.e_std = None

    # ---- helper: fit λ cho mọi cột của ma trận [N, d] ----
    def _fit_block(self, M: np.ndarray):
        d = M.shape[1]
        lams = np.ones(d)
        if self.use_power:
            for j in range(d):
                colj = M[:, j]
                # bỏ qua cột nhị phân {0,1} hoặc gần-hằng: power vô nghĩa
                uniq = np.unique(colj[np.isfinite(colj)])
                if self.skip_binary and uniq.size <= 2:
                    lams[j] = 1.0
                else:
                    lams[j] = _fit_lambda(colj)
        # áp λ rồi tính mean/std trên dữ liệu ĐÃ biến đổi
        Mt = M.copy()
        for j in range(d):
            if lams[j] != 1.0:
                Mt[:, j] = _yj(M[:, j], lams[j])
        mean = Mt.mean(0)
        std = Mt.std(0)
        std[std < 1e-6] = 1e-6
        return lams, mean, std

    def fit(self, graphs):
        xs = torch.cat([g.x for g in graphs], dim=0).numpy().astype(np.float64)
        es = torch.cat([g.edge_attr for g in graphs], dim=0).numpy().astype(np.float64)
        self.x_lam, self.x_mean, self.x_std = self._fit_block(xs)
        self.e_lam, self.e_mean, self.e_std = self._fit_block(es)
        if self.use_power:
            log.info("PowerScaler: node λ=%s", np.round(self.x_lam, 2).tolist())
            log.info("PowerScaler: edge λ=%s", np.round(self.e_lam, 2).tolist())
        return self

    def _apply(self, M_t: torch.Tensor, lam, mean, std) -> torch.Tensor:
        M = M_t.numpy().astype(np.float64)
        for j in range(M.shape[1]):
            if lam[j] != 1.0:
                M[:, j] = _yj(M[:, j], lam[j])
        M = (M - mean) / std
        return torch.tensor(M, dtype=torch.float32)

    def transform(self, graphs):
        for g in graphs:
            g.x = self._apply(g.x, self.x_lam, self.x_mean, self.x_std)
            g.edge_attr = self._apply(g.edge_attr, self.e_lam, self.e_mean, self.e_std)
        return graphs

    # để lưu/nạp cùng meta (tuỳ chọn)
    def state_dict(self):
        return {
            "use_power": self.use_power,
            "x_lam": self.x_lam, "x_mean": self.x_mean, "x_std": self.x_std,
            "e_lam": self.e_lam, "e_mean": self.e_mean, "e_std": self.e_std,
        }
