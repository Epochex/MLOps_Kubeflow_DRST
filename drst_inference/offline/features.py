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
    Prefer to read from MinIO (TRAIN_MINIO_KEY); otherwise read local file (TRAIN_LOCAL_PATH, default datasets/train.csv).
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

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def select_features(df: pd.DataFrame, topk: Optional[int] = None) -> List[str]:
    """
    Simple/robust: compute absolute Pearson correlation between numeric columns and TARGET, sort descending, pick topk;
    if topk not set, use all non-TARGET columns (excluding EXCLUDE_COLS).
    """
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
    Returns Xtr, Xva, Ytr, Yva, selected_feats, scaler.
    Side effect: save selected_feats.json + scaler.pkl to models/.
    """
    df = _read_training_table()
    df = _coerce_numeric(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    # Drop EXCLUDE_COLS
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df = df[cols]

    selected = select_features(df, topk=topk)

    X = df[selected].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Train/validation split
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

    # Save artefacts: selected_feats.json + scaler.pkl
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
