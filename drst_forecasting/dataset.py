# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/dataset.py
from __future__ import annotations
import io
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from drst_common.minio_helper import load_csv, save_bytes, s3, BUCKET
from drst_common.config import MODEL_DIR, DATA_DIR, TARGET_COL

def _load_selected_features() -> List[str]:
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
    feats = json.loads(raw.decode("utf-8"))
    return [str(c) for c in feats]

def _load_offline_scaler() -> StandardScaler:
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
    return joblib.load(io.BytesIO(raw))

def _save_forecast_scaler(sc: StandardScaler) -> None:
    bio = io.BytesIO()
    joblib.dump(sc, bio)
    save_bytes(f"{MODEL_DIR}/forecasting/scaler.pkl", bio.getvalue(), "application/octet-stream")

def build_sliding_window(
    lookback: int,
    horizon: int,
    take_last_n: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    返回：
      X: [N, lookback, D]  (多变量)
      y: [N, horizon]      (仅目标变量未来 H 步)
      feats: 特征名列表（与 X 的最后一维一致）
    """
    df = load_csv(f"{DATA_DIR}/combined.csv").reset_index(drop=True)

    # 选 offline 的 selected_feats（和训练一致），并使用 offline scaler 进行数值标准化
    feats = _load_selected_features()
    assert all(c in df.columns for c in feats), "selected_feats 中存在不在 combined.csv 的列"
    # 清理与数值化
    for c in feats + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats + [TARGET_COL]).reset_index(drop=True)

    if take_last_n and take_last_n > 0:
        df = df.tail(int(take_last_n)).reset_index(drop=True)

    # 用 offline scaler 对特征做 transform（预测目标不做缩放）
    sc_off = _load_offline_scaler()
    X_all = sc_off.transform(df[feats].astype(np.float32).values)     # [N, D]
    y_all = df[TARGET_COL].astype(np.float32).values                  # [N]

    T = int(lookback); H = int(horizon)
    xs, ys = [], []
    for i in range(0, len(df) - T - H + 1):
        hist = X_all[i:i+T, :]           # [T, D]
        future = y_all[i+T:i+T+H]        # [H]
        xs.append(hist)
        ys.append(future)
    if not xs:
        raise RuntimeError("数据不足以构成任何一个滑动窗口样本，请减小 lookback/horizon 或扩大数据。")
    X = np.stack(xs, axis=0).astype(np.float32)  # [N, T, D]
    Y = np.stack(ys, axis=0).astype(np.float32)  # [N, H]

    # forecasting 专用 scaler：仅对输入特征再 fit 一次（有必要时；通常复用 offline 的也可）
    sc_fore = StandardScaler().fit(X.reshape(X.shape[0], -1))
    _save_forecast_scaler(sc_fore)

    return X, Y, feats
