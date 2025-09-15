# drst_model_selection/common.py
from __future__ import annotations
import io, time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import save_bytes
from drst_common.config import RESULT_DIR

def rel_acc(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.05) -> float:
    yt = np.asarray(y_true, dtype=np.float32).ravel()
    yp = np.asarray(y_pred, dtype=np.float32).ravel()
    denom = np.maximum(np.abs(yt), 1e-8)
    return float((np.abs(yp - yt) / denom <= thr).mean())

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, per_sample_ms: float) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float32).ravel()
    yp = np.asarray(y_pred, dtype=np.float32).ravel()
    mae = float(np.mean(np.abs(yp - yt)))
    # R²：避免除零
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "r2": r2, "acc@5%": rel_acc(yt, yp, 0.05), "latency_ms": float(per_sample_ms)}

def bench_latency(predict_fn, X: np.ndarray, repeat: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        _ = predict_fn(X)
    dt = time.perf_counter() - t0
    return float(dt / max(1, repeat) / len(X) * 1000.0)

def save_rank_csv(name: str, df: pd.DataFrame) -> str:
    key = f"{RESULT_DIR}/forecasting/{name}"
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")
    return key
