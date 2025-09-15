# drst_model_selection/perf.py
from __future__ import annotations
import io, json, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
from sklearn.svm import SVR

from drst_common.minio_helper import load_csv, save_bytes
from drst_common.config import RESULT_DIR, MODEL_DIR

def _rel_acc(y_true, y_pred, thr=0.05):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), 1e-8)
    return float(np.mean(np.abs(yp - yt) / denom <= float(thr)))

def _bench_latency(fn, X):
    t0 = time.perf_counter(); _ = fn(X); dt = time.perf_counter() - t0
    return float(dt / len(X) * 1000.0)  # ms/sample

def run_perf_selection(
    perf_key: str = "datasets/perf/stage1_random_rates.csv",
    topk: int = 4,
    include_svr: bool = False,
    include_dt: bool = False,
):
    # 1) 取数据
    df = load_csv(perf_key).dropna()
    # 目标为 output_rate；去掉 input/output/latency 三列作为特征
    drop_cols = [c for c in ["input_rate", "output_rate", "latency"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").astype(float).values
    y = df["output_rate"].astype(float).values
    # 时间序切分：最后 30% 做测试
    n = len(df); v = max(1, int(0.3*n))
    X_train, X_test = X[:-v], X[-v:]
    y_train, y_test = y[:-v], y[-v:]

    # 2) 候选模型
    candidates = [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("mdl", Ridge(alpha=1.0))])),
        ("linear", Pipeline([("scaler", StandardScaler()), ("mdl", LinearRegression())])),
        ("rf",     RandomForestRegressor(n_estimators=300, max_depth=6, n_jobs=-1, random_state=0)),
        ("gbr",    GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=200, random_state=0)),
    ]
    if _HAS_XGB:
        candidates.append(("xgb", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                               subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                               reg_lambda=1.0, random_state=0)))
    if include_dt:
        candidates.append(("dt", DecisionTreeRegressor(max_depth=6, random_state=0)))
    if include_svr:
        candidates.append(("svr", Pipeline([("scaler", StandardScaler()),
                                            ("mdl", SVR(C=10.0, epsilon=0.1, gamma="scale"))])))

    # 3) 评测
    rows = []
    for name, mdl in candidates:
        t0 = time.perf_counter()
        mdl.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        y_pred = mdl.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))
        acc5 = _rel_acc(y_test, y_pred, 0.05)
        lat_ms = _bench_latency(lambda Z: mdl.predict(Z), X_test)

        rows.append({
            "model": name, "mae": round(mae,6), "r2": round(r2,6),
            "acc@5%": round(acc5,6), "latency_ms": round(lat_ms,6),
            "fit_seconds": round(fit_s, 6)
        })

    df_out = pd.DataFrame(rows).sort_values(["mae","latency_ms"], ascending=[True, True])
    # 4) 写结果
    csv = df_out.to_csv(index=False).encode("utf-8")
    save_bytes(f"{RESULT_DIR}/perf/model_selection_results.csv", csv, "text/csv")
    top = df_out.head(max(1, int(topk))).to_dict(orient="records")
    save_bytes(f"{MODEL_DIR}/perf/selection.json", json.dumps({"top": top}, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
    print("[perf.model_selection] wrote:",
          f"s3://.../{RESULT_DIR}/perf/model_selection_results.csv and s3://.../{MODEL_DIR}/perf/selection.json",
          flush=True)
