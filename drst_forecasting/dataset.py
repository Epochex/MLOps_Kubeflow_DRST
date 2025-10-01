# drst_forecasting/dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, BUCKET, load_csv, save_bytes
from drst_common.config import MODEL_DIR

PCM_FULL_KEY      = os.getenv("PCM_FULL_KEY", "datasets/pcm/pcm_global.csv")
SELECTED_FEATS_KEY = f"{MODEL_DIR}/selected_feats.json"
TARGET_COL        = os.getenv("FORECAST_TARGET", "latency")   # default target column for forecasting

def _save_selected_features(feats: List[str]) -> None:
    buf = json.dumps(feats).encode("utf-8")
    save_bytes(SELECTED_FEATS_KEY, buf, "application/json")

def _load_selected_features() -> List[str]:
    """Load models/selected_feats.json if available; otherwise infer from the full PCM table and save."""
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=SELECTED_FEATS_KEY)["Body"].read()
        feats = json.loads(raw.decode("utf-8"))
        if isinstance(feats, list) and feats:
            return [str(c) for c in feats]
    except Exception:
        pass

    # Fallback: infer from the merged PCM dataset
    df = load_csv(PCM_FULL_KEY)
    # Select numeric columns, excluding the target
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in num_cols if c != TARGET_COL]
    if not feats:
        raise RuntimeError(f"fallback feature discovery failed: no numeric columns (key=s3://{BUCKET}/{PCM_FULL_KEY})")

    # Save for future use
    _save_selected_features(feats)
    return feats

def _load_series() -> pd.DataFrame:
    """Load the merged PCM dataset and perform basic cleaning on key columns."""
    df = load_csv(PCM_FULL_KEY)
    # Force numeric conversion (tolerant mode)
    for c in df.columns:
        if c == TARGET_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif pd.api.types.is_object_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    # Drop rows where the target is missing
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"target column '{TARGET_COL}' not found in PCM dataset (key={PCM_FULL_KEY})")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df

def build_sliding_window(
    lookback: int,
    horizon: int,
    take_last_n: int | None = None,
    multi_output: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    feats = _load_selected_features()
    df = _load_series()

    # Drop rows with missing feature values; fill remaining NaNs with 0
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    sub = df[feats + [TARGET_COL]].copy()
    sub[feats] = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sub[TARGET_COL] = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    sub = sub.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X_list = []
    Y_list = []
    values = sub[feats].values
    target = sub[TARGET_COL].values
    L = len(sub)

    # Generate sliding windows
    end = L - lookback - horizon + 1
    for i in range(max(0, end)):
        X_list.append(values[i:i+lookback, :])
        if multi_output:
            # Predict the next `horizon` points
            Y_list.append(target[i+lookback : i+lookback+horizon])
        else:
            # Predict only y_{t+H}
            Y_list.append(target[i+lookback+horizon-1])

    if not X_list:
        raise RuntimeError(f"not enough rows to build sliding windows: rows={L}, lookback={lookback}, horizon={horizon}")

    X = np.stack(X_list, axis=0)
    if multi_output:
        Y = np.stack(Y_list, axis=0).astype(float)  # (N, H)
    else:
        Y = np.asarray(Y_list, dtype=float)         # (N,)

    if take_last_n and take_last_n > 0:
        X = X[-take_last_n:, :, :]
        Y = Y[-take_last_n:]

    return X, Y, feats