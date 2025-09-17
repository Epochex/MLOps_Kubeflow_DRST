#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabular Perf — Hyperparameter Search

- 读取 MinIO 上的 perf 数据集（默认 datasets/combined.csv）
- 目标列：output_rate（若 config.TARGET_COL 指定了别名则以其为准）
- 候选模型：Ridge / RandomForest / GradientBoosting / XGBoost(可选)
- 指标：MAE / R^2 / acc@5% / 每样本推理延时
- 产物：
  - models/forecasting/perf_hpsearch_results.csv        （全量结果）
  - models/forecasting/perf_hpsearch_best.json          （最优配置建议）

注意：
- 这是“表格回归”的 HP 搜索，和 PCM 的时序 HP 搜索相互独立。
- 仅写入建议与结果清单，不会改动 latest 指针（由 retrain 决定）。
"""

from __future__ import annotations
import io
import os
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# XGBoost 可选
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from drst_common.minio_helper import load_csv, save_bytes
from drst_common.config import MODEL_DIR, TARGET_COL

# --------- 工具 ---------
def _safe_num_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, per_sample_ms: float) -> Dict[str, Any]:
    yt = np.asarray(y_true, np.float32).ravel()
    yp = np.asarray(y_pred, np.float32).ravel()
    mae = float(np.mean(np.abs(yp - yt)))
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    denom = np.maximum(np.abs(yt), 1e-8)
    acc5 = float((np.abs(yp - yt) / denom <= 0.05).mean())
    return {"mae": mae, "r2": r2, "acc@5%": acc5, "latency_ms": float(per_sample_ms)}

def _bench_latency(predict_fn, X: np.ndarray, repeat: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        _ = predict_fn(X)
    dt = time.perf_counter() - t0
    return float(dt / max(1, repeat) / len(X) * 1000.0)

def _save_csv(key: str, df: pd.DataFrame) -> None:
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

# --------- 主流程 ---------
def main():
    ap = argparse.ArgumentParser("perf hpsearch")
    ap.add_argument("--data_path", type=str, default="datasets/combined.csv")
    ap.add_argument("--n_jobs", type=int, default=0)
    args = ap.parse_args()

    print(f"[perf.hpsearch] input=s3://.../{args.data_path}")
    df = load_csv(args.data_path)
    df = _safe_num_df(df).dropna(how="any").reset_index(drop=True)

    # 选择特征与目标（保守地剔除易泄漏/非数值）
    drop_cols = {"input_rate", "latency"}
    target = TARGET_COL if TARGET_COL in df.columns else "output_rate"
    if target not in df.columns:
        raise RuntimeError(f"Target column '{target}' not found in {args.data_path}")

    feats = [c for c in df.columns if c not in drop_cols.union({target})]
    X = df[feats].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)

    rows: List[Dict[str, Any]] = []

    # —— Linear baseline （做个垫底参考）——
    lin = Pipeline([("sc", StandardScaler(with_mean=False)), ("lr", LinearRegression())])
    lin.fit(Xtr, ytr)
    yp = lin.predict(Xva)
    lat = _bench_latency(lambda Z: lin.predict(Z), Xva, repeat=1)
    ev = _evaluate(yva, yp, lat); ev.update({"model": "Linear"})
    rows.append(ev)

    # —— Ridge —— 
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        rg = Pipeline([("sc", StandardScaler(with_mean=False)), ("rg", Ridge(alpha=alpha))])
        rg.fit(Xtr, ytr)
        yp = rg.predict(Xva)
        lat = _bench_latency(lambda Z: rg.predict(Z), Xva, repeat=1)
        ev = _evaluate(yva, yp, lat); ev.update({"model": "Ridge", "alpha": alpha})
        rows.append(ev)

    # —— RandomForest —— 
    for n in [200, 400, 800]:
        for d in [6, 8, 10]:
            rf = RandomForestRegressor(
                n_estimators=n, max_depth=d, min_samples_leaf=5, random_state=0,
                n_jobs=(None if args.n_jobs <= 0 else args.n_jobs)
            )
            rf.fit(Xtr, ytr)
            yp = rf.predict(Xva)
            lat = _bench_latency(lambda Z: rf.predict(Z), Xva, repeat=1)
            ev = _evaluate(yva, yp, lat); ev.update({"model": "RandomForest", "n_estimators": n, "max_depth": d})
            rows.append(ev)

    # —— GradientBoosting —— 
    for n in [200, 400, 800]:
        for lr in [0.1, 0.05]:
            gb = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=3, random_state=0)
            gb.fit(Xtr, ytr)
            yp = gb.predict(Xva)
            lat = _bench_latency(lambda Z: gb.predict(Z), Xva, repeat=1)
            ev = _evaluate(yva, yp, lat); ev.update({"model": "GradientBoosting", "n_estimators": n, "learning_rate": lr})
            rows.append(ev)

    # —— XGBoost（可选）——
    if _HAS_XGB:
        for n in [200, 400, 800]:
            for d in [6, 8]:
                for lr in [0.1, 0.05]:
                    xgb = XGBRegressor(
                        n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, reg_alpha=0.0, objective="reg:squarederror",
                        tree_method="hist", n_jobs=(None if args.n_jobs <= 0 else args.n_jobs)
                    )
                    xgb.fit(Xtr, ytr)
                    yp = xgb.predict(Xva)
                    lat = _bench_latency(lambda Z: xgb.predict(Z), Xva, repeat=1)
                    ev = _evaluate(yva, yp, lat); ev.update({"model": "XGBoost", "n_estimators": n, "max_depth": d, "learning_rate": lr})
                    rows.append(ev)
    else:
        print("[perf.hpsearch] XGBoost not available — skipped.")

    # —— 排序/落盘 —— 
    df_res = pd.DataFrame(rows).sort_values(["mae", "latency_ms"]).reset_index(drop=True)
    csv_key = f"{MODEL_DIR}/forecasting/perf_hpsearch_results.csv"
    _save_csv(csv_key, df_res)
    print(f"[perf.hpsearch] wrote results -> s3://.../{csv_key}")

    best = df_res.iloc[0].to_dict()
    best_json = {
        "task": "perf",
        "winner": str(best.get("model")),
        "metrics": {
            "mae": float(best.get("mae", float("nan"))),
            "r2": float(best.get("r2", float("nan"))),
            "acc@5%": float(best.get("acc@5%", float("nan"))),
            "latency_ms": float(best.get("latency_ms", float("nan"))),
        },
        "note": "Use as a prior for offline training / monitoring dashboards.",
    }
    save_bytes(f"{MODEL_DIR}/forecasting/perf_hpsearch_best.json",
               json.dumps(best_json, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
    print(f"[perf.hpsearch] best = {json.dumps(best_json, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
