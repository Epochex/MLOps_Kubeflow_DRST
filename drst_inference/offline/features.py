#!/usr/bin/env python3
# drst_inference/offline/features.py
from __future__ import annotations
import io
import os
import json
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from drst_common.config import (
    FEATURE_SRC_KEY, EXCLUDE_COLS, TARGET_COL, MODEL_DIR
)
from drst_common.minio_helper import load_csv, save_bytes
from drst_common.metric_logger import log_metric

EPS_STD = 1e-6  # 判定“近似常数列”的阈值

# ----------------------------------------------------------------------
# 1) 从 FEATURE_SRC_KEY 派生【候选特征集合】（唯一真理源）
# ----------------------------------------------------------------------
def derive_feature_schema() -> List[str]:
    df_src = load_csv(FEATURE_SRC_KEY)
    # 清洗：去噪/去空/去非数值；排除显式列与目标列
    cols = [c for c in df_src.columns if c not in set(EXCLUDE_COLS + [TARGET_COL])]
    num = df_src[cols].apply(pd.to_numeric, errors="coerce")
    num = num.dropna(how="all", axis=1)  # 去掉完全空列
    feats = list(num.columns)
    if not feats:
        raise ValueError(f"[features] No candidate features found from {FEATURE_SRC_KEY}")
    return feats

# ----------------------------------------------------------------------
# 2) 对任意训练表做 schema 自检
# ----------------------------------------------------------------------
def schema_check(df: pd.DataFrame, schema_feats: List[str]) -> Dict[str, List[str]]:
    present = [c for c in schema_feats if c in df.columns]
    missing = [c for c in schema_feats if c not in df.columns]
    extra   = [c for c in df.columns if c not in set(schema_feats + [TARGET_COL] + EXCLUDE_COLS)]
    zero_only = []
    if present:
        num = df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        std = num.std(axis=0)
        zero_only = [c for c in present if float(std[c]) <= EPS_STD]
    return {"present": present, "missing": missing, "extra": extra, "zero_only": zero_only}

def print_schema_check(tag: str, df: pd.DataFrame, schema_feats: List[str]) -> None:
    info = schema_check(df, schema_feats)
    print(f"[features] schema-check@{tag}: "
          f"present={len(info['present'])}/{len(schema_feats)}, "
          f"missing={len(info['missing'])}, zero-only={len(info['zero_only'])}, extra={len(info['extra'])}")
    if info["missing"][:5]:
        print(f"[features]   missing(sample) -> {info['missing'][:5]}")
    if info["zero_only"][:5]:
        print(f"[features]   zero-only(sample) -> {info['zero_only'][:5]}")

# ----------------------------------------------------------------------
# 3) 内部：读取训练表（MinIO 优先；支持本地回退）
#    - 与 train_offline.py 保持一致：环境变量 TRAIN_MINIO_KEY / TRAIN_LOCAL_PATH
# ----------------------------------------------------------------------
def _read_training_table() -> pd.DataFrame:
    key = os.getenv("TRAIN_MINIO_KEY", "").strip()
    if key:
        return load_csv(key)
    path = os.getenv("TRAIN_LOCAL_PATH", "datasets/train.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

# ----------------------------------------------------------------------
# 4) 公开方法：准备数据（含 schema 自检 + 相关性选前 topk + 写工件）
#    返回 Xtr, Xva, Ytr, Yva, selected_feats, scaler
# ----------------------------------------------------------------------
def prepare_dataset(topk: Optional[int] = None, val_frac: float = 0.2, seed: int = 42
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    # 4.1 候选特征集合（唯一真理源）
    schema_feats = derive_feature_schema()

    # 4.2 读取训练全集 + 初步清洗
    df = _read_training_table()
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 目标列必须有值
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 4.3 schema 自检（日志）
    print_schema_check("train", df, schema_feats)

    # 4.4 只使用 schema 中的列；对缺失列补 0（保证线上/线下维度一致）
    for c in schema_feats:
        if c not in df.columns:
            df[c] = 0.0
    df = df[schema_feats + [TARGET_COL]]

    # 4.4b 特征清洗：把 ±Inf -> NaN，再用 0 填充 NaN，避免 scaler/训练出现 NaN
    df[schema_feats] = (
        df[schema_feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 4.5 剔除近似常数列（避免全 0 影响相关性排序）
    std = df[schema_feats].std(axis=0)
    good = [c for c in schema_feats if float(std[c]) > EPS_STD]
    if not good:
        raise ValueError("[features] all candidate features are (near-)constant after cleaning")
    if len(good) < len(schema_feats):
        drop_n = len(schema_feats) - len(good)
        print(f"[features] drop ~constant columns: {drop_n} / {len(schema_feats)}")

    # 4.6 相关性选前 topk（只在 good 子集中排序）
    corr = df[good].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
    order = list(corr.index)
    if topk is not None and topk > 0:
        order = order[:topk]
    selected = order

    # 4.7 划分 & 标准化
    X = df[selected].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    val_n = max(1, int(len(X) * val_frac))
    va_idx, tr_idx = idx[:val_n], idx[val_n:]

    scaler = StandardScaler().fit(X[tr_idx])
    Xtr = scaler.transform(X[tr_idx])
    Xva = scaler.transform(X[va_idx])
    Ytr, Yva = y[tr_idx], y[va_idx]

    # 4.8 持久化工件：selected_feats.json + scaler.pkl
    save_bytes(
        f"{MODEL_DIR}/selected_feats.json",
        json.dumps(selected, ensure_ascii=False, indent=2).encode(),
        "application/json"
    )

    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    save_bytes(f"{MODEL_DIR}/scaler.pkl", buf.getvalue(), "application/octet-stream")

    log_metric(
        component="offline", event="features_ready",
        train_rows=int(len(tr_idx)), val_rows=int(len(va_idx)),
        schema_cols=len(schema_feats), selected_cols=len(selected)
    )

    return Xtr, Xva, Ytr, Yva, selected, scaler
