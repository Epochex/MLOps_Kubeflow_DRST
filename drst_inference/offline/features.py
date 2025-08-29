#!/usr/bin/env python3
# drst_inference/offline/features.py
from __future__ import annotations
import io
import os
import json
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from drst_common.config import EXCLUDE_COLS, TARGET_COL, MODEL_DIR
from drst_common.minio_helper import load_csv, save_bytes
from drst_common.metric_logger import log_metric

def _read_training_table() -> pd.DataFrame:
    """
    优先从 MinIO 读取（TRAIN_MINIO_KEY），否则读本地 CSV（TRAIN_LOCAL_PATH，默认 datasets/train.csv）。
    """
    key = os.getenv("TRAIN_MINIO_KEY", "").strip()
    if key:
        df = load_csv(key)
    else:
        path = os.getenv("TRAIN_LOCAL_PATH", "datasets/train.csv")
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
    return df

def _restrict_to_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    若设置 FEATURE_COLS（逗号分隔），则仅保留该集合交集；否则返回原 df。
    """
    raw = os.getenv("FEATURE_COLS", "").strip()
    if not raw:
        return df
    allow = [c.strip() for c in raw.split(",") if c.strip()]
    keep = [c for c in df.columns if c in allow]
    if not keep:
        return df
    return df[keep + ([TARGET_COL] if TARGET_COL in df.columns else [])]

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def select_features(df: pd.DataFrame, topk: Optional[int] = None) -> List[str]:
    """
    绝对 Pearson 相关性：对候选数值列与 TARGET 求 |corr|，降序取前 topk。
    若未设 topk，使用全部候选列。
    """
    # 候选：排除 EXCLUDE_COLS 与 TARGET
    cand = [c for c in df.columns if c not in set(EXCLUDE_COLS + [TARGET_COL])]
    num = df[cand].select_dtypes(include=["number"])
    corr = num.corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
    order = list(corr.index)
    if topk is not None and topk > 0:
        order = order[:topk]
    return order

def prepare_dataset(topk: Optional[int] = None, val_frac: float = 0.2, seed: int = 42
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    返回 Xtr, Xva, Ytr, Yva, selected_feats, scaler。
    副作用：将 selected_feats.json 与 scaler.pkl 写入 models/ 供在线侧热加载。
    """
    df = _read_training_table()
    # 可选限制在 FEATURE_COLS 集合
    df = _restrict_to_feature_cols(df)
    df = _coerce_numeric(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 丢弃无意义/高泄露列（配置中已包含 input_rate/latency 等）
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df = df[cols]

    selected = select_features(df, topk=topk)

    X = df[selected].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # 训练/验证划分
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    val_n = max(1, int(len(X) * val_frac))
    va_idx, tr_idx = idx[:val_n], idx[val_n:]

    scaler = StandardScaler().fit(X[tr_idx])
    Xtr = scaler.transform(X[tr_idx])
    Xva = scaler.transform(X[va_idx])

    Ytr = y[tr_idx]
    Yva = y[va_idx]

    # 持久化工件：selected_feats.json + scaler.pkl
    save_bytes(f"{MODEL_DIR}/selected_feats.json",
               json.dumps(selected, ensure_ascii=False, indent=2).encode(),
               "application/json")
    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    save_bytes(f"{MODEL_DIR}/scaler.pkl", buf.getvalue(), "application/octet-stream")

    log_metric(component="offline", event="features_ready",
               train_rows=int(len(tr_idx)), val_rows=int(len(va_idx)),
               value=len(selected))

    return Xtr, Xva, Ytr, Yva, selected, scaler
