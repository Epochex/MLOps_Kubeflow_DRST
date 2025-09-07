#!/usr/bin/env python3
# drst_common/utils.py
from __future__ import annotations
import numpy as np

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return np.array(x, dtype=float)

def calculate_accuracy_within_threshold(
    y_true,
    y_pred,
    thr: float,
    *,
    mode: str = "relative",
    eps: float = 1e-6,
) -> float:
    """
    返回百分比 (0~100)。
    mode="relative": |pred - true| / max(|true|, eps) <= thr
    自动屏蔽 nan/inf；避免 true≈0 时相对误差被放大。
    """
    yt = _to_numpy(y_true).astype(np.float64)
    yp = _to_numpy(y_pred).astype(np.float64)

    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return 0.0
    yt = yt[m]; yp = yp[m]

    err = np.abs(yp - yt)
    if str(mode).lower().startswith("rel"):
        denom = np.maximum(np.abs(yt), float(eps))
        rel = err / denom
        ok = rel <= float(thr)
    else:
        ok = err <= float(thr)

    acc = 100.0 * (np.sum(ok) / ok.size)
    return float(acc)

def make_prob_hist(x: np.ndarray, bins: int = 64, range=None) -> np.ndarray:
    x = _to_numpy(x).ravel()
    if range is not None:
        hist, _ = np.histogram(x, bins=bins, range=range)
    else:
        hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.ones(bins, dtype=np.float64) / float(bins)
    return p / s

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _to_numpy(p).astype(np.float64)
    q = _to_numpy(q).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
