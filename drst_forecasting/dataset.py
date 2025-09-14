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

# ---- 可配置（也可用环境变量覆盖）----
PCM_FULL_KEY      = os.getenv("PCM_FULL_KEY", "datasets/pcm/pcm_global.csv")
SELECTED_FEATS_KEY = f"{MODEL_DIR}/selected_feats.json"
TARGET_COL        = os.getenv("FORECAST_TARGET", "latency")   # 默认用 latency 做时序预测目标

def _save_selected_features(feats: List[str]) -> None:
    buf = json.dumps(feats).encode("utf-8")
    save_bytes(SELECTED_FEATS_KEY, buf, "application/json")

def _load_selected_features() -> List[str]:
    """优先读 models/selected_feats.json；若不存在，就从 PCM 全量表自动推断并保存。"""
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=SELECTED_FEATS_KEY)["Body"].read()
        feats = json.loads(raw.decode("utf-8"))
        if isinstance(feats, list) and feats:
            return [str(c) for c in feats]
    except Exception:
        pass

    # 回退：从 PCM 合并集推断
    df = load_csv(PCM_FULL_KEY)
    # 选全部数值列，去掉目标列
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in num_cols if c != TARGET_COL]
    if not feats:
        raise RuntimeError(f"fallback feature discovery failed: no numeric columns (key=s3://{BUCKET}/{PCM_FULL_KEY})")

    # 保存供下次使用
    _save_selected_features(feats)
    return feats

def _load_series() -> pd.DataFrame:
    """载入 PCM 合并表；对关键列做基本清洗。"""
    df = load_csv(PCM_FULL_KEY)
    # 强制数值化（容错）
    for c in df.columns:
        if c == TARGET_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif pd.api.types.is_object_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    # 去掉目标缺失的行
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"target column '{TARGET_COL}' not found in PCM dataset (key={PCM_FULL_KEY})")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df

def build_sliding_window(lookback: int, horizon: int, take_last_n: int | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    用 PCM 合并表构造滑窗数据：
      X shape: (N, lookback, F)
      Y shape: (N, )
    """
    feats = _load_selected_features()
    df = _load_series()

    # 丢弃缺失的特征行；对剩余缺失值用 0 补（时序建模常见做法，你也可以改成前向填充）
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

    # 生成窗口
    end = L - lookback - horizon + 1
    for i in range(max(0, end)):
        X_list.append(values[i:i+lookback, :])
        Y_list.append(target[i+lookback+horizon-1])

    if not X_list:
        raise RuntimeError(f"not enough rows to build sliding windows: rows={L}, lookback={lookback}, horizon={horizon}")

    X = np.stack(X_list, axis=0)
    Y = np.asarray(Y_list, dtype=float)

    if take_last_n and take_last_n > 0:
        X = X[-take_last_n:, :, :]
        Y = Y[-take_last_n:]

    return X, Y, feats
