#!/usr/bin/env python3
# drst_inference/offline/features.py
from __future__ import annotations
import io
import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from drst_common.minio_helper import load_csv, save_bytes
from drst_common.config import MODEL_DIR, TARGET_COL

EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "latency"]

def _clean_values(df: pd.DataFrame) -> pd.DataFrame:
    # Replace placeholders/blank strings with NaN
    df = df.replace({"<not counted>": np.nan, r"^\s*$": np.nan}, regex=True)
    return df

def _derive_feature_cols_from_combined(off_key: str) -> List[str]:
    # Discover usable numeric feature columns from the combined dataset
    df_raw = load_csv(off_key)
    df_raw = _clean_values(df_raw)
    cand_cols = [c for c in df_raw.columns if c not in set(EXCLUDE_COLS + [TARGET_COL])]
    num = df_raw[cand_cols].apply(pd.to_numeric, errors="coerce")
    feat_cols = [c for c in cand_cols if not num[c].isna().all()]
    if len(feat_cols) == 0:
        raise RuntimeError(f"No numeric feature columns derived from {off_key}")
    save_bytes(
        f"{MODEL_DIR}/feature_cols.json",
        json.dumps(feat_cols, ensure_ascii=False, indent=2).encode(),
        "application/json",
    )
    return feat_cols

def _read_clean(off_key: str, feature_cols: List[str]) -> pd.DataFrame:
    # Load combined dataset, clean, align columns, and ensure numeric types
    df = load_csv(off_key)
    df = _clean_values(df)
    df = df.dropna(how="any").reset_index(drop=True)
    df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors="ignore")
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = np.nan
    keep_cols = feature_cols + [TARGET_COL]
    df = df[keep_cols]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df

def select_topk_features(df_all: pd.DataFrame, feature_cols: List[str], k: int = 10) -> List[str]:
    # Rank features by |Pearson correlation| with the target and take top-k (pad if fewer)
    corr = df_all[feature_cols].corrwith(df_all[TARGET_COL]).abs().fillna(0.0).sort_values(ascending=False)
    topk = list(corr.index[:int(k)])
    if len(topk) < k:
        remain = [c for c in feature_cols if c not in topk]
        topk += remain[:(k - len(topk))]
    save_bytes(
        f"{MODEL_DIR}/selected_feats.json",
        json.dumps(topk, ensure_ascii=False, indent=2).encode(),
        "application/json",
    )
    return topk

def fit_scaler_on_all(df_all: pd.DataFrame, selected: List[str]) -> StandardScaler:
    # Fit a StandardScaler on all rows/selected columns and persist it
    X = df_all[selected].astype(np.float32).values
    scaler = StandardScaler().fit(X)
    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    save_bytes(f"{MODEL_DIR}/scaler.pkl", buf.getvalue(), "application/octet-stream")
    return scaler

def load_and_prepare(off_key: str, k: int = 10) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Lazy loading: only read the combined dataset (off_key) from MinIO when called.

    Steps:
      1) Bootstrap the full set of usable numeric features from the combined dataset → feature_cols
         (also written to feature_cols.json).
      2) Clean and align columns using feature_cols to produce df_all (includes TARGET_COL).
      3) Compute top-k by absolute Pearson correlation → selected_feats.json.
      4) Fit a StandardScaler on df_all[selected] → scaler.pkl.
    """
    feature_cols = _derive_feature_cols_from_combined(off_key)
    df_all = _read_clean(off_key, feature_cols)
    selected = select_topk_features(df_all, feature_cols, k=k)
    scaler = fit_scaler_on_all(df_all, selected)
    return df_all, selected, scaler
