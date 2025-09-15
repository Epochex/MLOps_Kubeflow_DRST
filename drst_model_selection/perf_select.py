# drst_model_selection/perf_select.py
from __future__ import annotations
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# XGBoost 可选
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from drst_common.minio_helper import load_csv, save_bytes
from drst_common.config import MODEL_DIR
from .common import evaluate, bench_latency, save_rank_csv

def _safe_num_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def run_perf_selection(
    perf_key: str = "datasets/perf/stage1_random_rates.csv",
    topk: int = 3,
    include_svr: bool = False,
    include_dt: bool = False,
) -> None:
    """
    Perf：表格回归，单输出（output_rate）。
    候选默认：Linear / Ridge / RandomForest / GradientBoosting / XGBoost（可选）
    可选：SVR / DecisionTree
    """
    print(f"[perf_select] input=s3://.../{perf_key}", flush=True)
    df = load_csv(perf_key)
    df = _safe_num_df(df).dropna(axis=0, how="any").reset_index(drop=True)

    drop_cols = ["input_rate", "output_rate", "latency"]
    feats = [c for c in df.columns if c not in drop_cols]
    X = df[feats].values.astype(np.float32)
    y = df["output_rate"].values.astype(np.float32)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)

    rows: List[Dict[str, Any]] = []

    # --- Linear ---
    lin = Pipeline([("sc", StandardScaler(with_mean=False)), ("lr", LinearRegression())])
    lin.fit(Xtr, ytr)
    yp = lin.predict(Xva)
    lat = bench_latency(lambda Z: lin.predict(Z), Xva, repeat=1)
    ev = evaluate(yva, yp, lat); ev.update({"model": "Linear"})
    rows.append(ev)

    # --- Ridge ---
    for a in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        rg = Pipeline([("sc", StandardScaler(with_mean=False)), ("rg", Ridge(alpha=a))])
        rg.fit(Xtr, ytr)
        yp = rg.predict(Xva)
        lat = bench_latency(lambda Z: rg.predict(Z), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "Ridge", "alpha": a})
        rows.append(ev)

    # --- RandomForest ---
    for n in [200, 400]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=-1)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xva)
        lat = bench_latency(lambda Z: rf.predict(Z), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "RandomForest", "n_estimators": n})
        rows.append(ev)

    # --- GradientBoosting ---
    for n in [200, 400]:
        gb = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, max_depth=3, random_state=0)
        gb.fit(Xtr, ytr)
        yp = gb.predict(Xva)
        lat = bench_latency(lambda Z: gb.predict(Z), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "GradientBoosting", "n_estimators": n})
        rows.append(ev)

    # --- XGBoost（可选） ---
    if _HAS_XGB:
        for n in [200, 400]:
            xgb = XGBRegressor(
                n_estimators=n, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                objective="reg:squarederror", tree_method="hist", n_jobs=0
            )
            xgb.fit(Xtr, ytr)
            yp = xgb.predict(Xva)
            lat = bench_latency(lambda Z: xgb.predict(Z), Xva, repeat=1)
            ev = evaluate(yva, yp, lat); ev.update({"model": "XGBoost", "n_estimators": n})
            rows.append(ev)

    # --- 可选补充 ---
    if include_dt:
        dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=0)
        dt.fit(Xtr, ytr)
        yp = dt.predict(Xva)
        lat = bench_latency(lambda Z: dt.predict(Z), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "DecisionTree"})
        rows.append(ev)

    if include_svr:
        svr = Pipeline([("sc", StandardScaler()), ("svr", SVR(C=10.0, epsilon=0.1, gamma="scale"))])
        svr.fit(Xtr, ytr)
        yp = svr.predict(Xva)
        lat = bench_latency(lambda Z: svr.predict(Z), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "SVR"})
        rows.append(ev)

    # 排序/保存
    df_rank = pd.DataFrame(rows).sort_values(["mae", "latency_ms"]).reset_index(drop=True)
    csv_key = save_rank_csv("perf_model_selection.csv", df_rank)
    print(f"[perf_select] wrote rank -> s3://.../{csv_key}", flush=True)

    # 建议清单（不覆盖 selected.json）
    best = df_rank.iloc[0].to_dict()
    suggestion = {
        "task": "perf",
        "features": feats,
        "recommend_topk": json.loads(df_rank.head(min(len(df_rank), max(1, int(topk)))).to_json(orient="records")),
        "winner": best.get("model"),
        "note": "Tabular single-output selection; use as a down-selection before HP search."
    }
    save_bytes(f"{MODEL_DIR}/forecasting/perf_select_suggestion.json",
               json.dumps(suggestion, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
    print(f"[perf_select] winner={best.get('model')} mae={best.get('mae'):.4f}", flush=True)
