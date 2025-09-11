# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/metrics.py
from __future__ import annotations
import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_pred - y_true)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def acc_within_threshold(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.maximum(np.abs(y_true), 1e-8)
    rel = np.abs(y_pred - y_true) / denom
    return float((rel <= float(thr)).mean() * 100.0)
