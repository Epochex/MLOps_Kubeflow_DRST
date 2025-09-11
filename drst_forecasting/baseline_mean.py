# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/baseline_mean.py
from __future__ import annotations
import numpy as np

def moving_mean_baseline(y_hist: np.ndarray, horizon: int) -> np.ndarray:
    """
    y_hist: [B, lookback] 仅目标序列的历史窗口
    return: [B, horizon]   用 lookback 平均值复制 horizon 次
    """
    y_hist = np.asarray(y_hist, dtype=np.float32)
    mu = np.mean(y_hist, axis=1, keepdims=True)   # [B,1]
    return np.repeat(mu, int(horizon), axis=1)
