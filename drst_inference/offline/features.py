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
    FEATURE_SRC_KEY, EXCLUDE_COLS, TARGET_COL, MODEL_DIR, OFFLINE_TOPK
)
from drst_common.minio_helper import load_csv, save_bytes
from drst_common.metric_logger import log_metric

EPS_STD = 1e-6  # 判定“近似常数列”的阈值

# ----------------------------------------------------------------------
# 1) 从 FEATURE_SRC_KEY 派生【候选特征集合】（唯一真理源，期望为 60 维）
#    注意：只排除 EXCLUDE_COLS + TARGET_COL，不做“近似常数列”剔除，
#    这样用于漂移监控的 FEATURE_COLS 就是完整 60 维。
# ----------------------------------------------------------------------
def derive_feature_schema() -> List[str]:
    df_src = load_csv(FEATURE_SRC_KEY)
    cols = [c for c in df_src.columns if c not in set(EXCLUDE_COLS + [TARGET_COL])]
    num = df_src[cols].apply(pd.to_numeric, errors="coerce")
    num = num.dropna(how="all", axis=1)  # 去掉完全空列（异常数据）
    feats = list(num.columns)
    if not feats:
        raise ValueError(f"[features] No candidate features found from {FEATURE_SRC_KEY}")
    return feats  # 作为 FEATURE_COLS（期望 60 维）

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
# ----------------------------------------------------------------------
def _read_training_table() -> pd.DataFrame:
    # 训练数据使用 OFFLINE_TRAIN_KEY（统一由 pipeline/调用方设置到 config）
    from drst_common.config import OFFLINE_TRAIN_KEY
    df = load_csv(OFFLINE_TRAIN_KEY)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

# ----------------------------------------------------------------------
# 4) 准备数据（schema 自检 + 相关性选前 topk + 持久化工件）
#    返回 Xtr, Xva, Ytr, Yva, selected_feats, scaler
# ----------------------------------------------------------------------
def prepare_dataset(topk: Optional[int] = None, val_frac: float = 0.2, seed: int = 42
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    # 4.1 候选特征集合（FEATURE_COLS，完整 60 维）
    schema_feats = derive_feature_schema()

    # —— 新增：把 FEATURE_COLS 落盘，供 producer / monitor 使用 —— #
    save_bytes(
        f"{MODEL_DIR}/feature_cols.json",
        json.dumps(schema_feats, ensure_ascii=False, indent=2).encode(),
        "application/json"
    )

    # 4.2 读取训练全集 + 初步清洗
    df = _read_training_table()
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 目标列必须有值
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 4.3 schema 自检（日志）
    print_schema_check("train", df, schema_feats)

    # 4.4 保证训练使用 schema 中的列；缺失列补 0（保证线上/线下维度一致）
    for c in schema_feats:
        if c not in df.columns:
            df[c] = 0.0
    df = df[schema_feats + [TARGET_COL]]

    # 4.4b 特征清洗：把 ±Inf -> NaN，再用 0 填充 NaN
    df[schema_feats] = (
        df[schema_feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 4.5 仅用于“训练特征选择”的近似常数列剔除（不影响上面的 FEATURE_COLS）
    std = df[schema_feats].std(axis=0)
    good = [c for c in schema_feats if float(std[c]) > EPS_STD]
    if not good:
        raise ValueError("[features] all candidate features are (near-)constant after cleaning")
    if len(good) < len(schema_feats):
        drop_n = len(schema_feats) - len(good)
        print(f"[features] drop ~constant columns for selection: {drop_n} / {len(schema_feats)}")

    # 4.6 与目标列的皮尔逊相关系数（绝对值）排序，选前 topk
    corr = df[good].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
    order = list(corr.index)
    k = int(topk if (topk is not None and topk > 0) else OFFLINE_TOPK)
    selected = order[:k]

    # 4.7 划分 & 标准化（仅针对 selected）
    X_all = df[selected].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_all))
    rng.shuffle(idx)
    val_n = max(1, int(len(X_all) * val_frac))
    va_idx, tr_idx = idx[:val_n], idx[val_n:]

    scaler = StandardScaler().fit(X_all[tr_idx])
    Xtr = scaler.transform(X_all[tr_idx])
    Xva = scaler.transform(X_all[va_idx])
    Ytr, Yva = y_all[tr_idx], y_all[va_idx]

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
