# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/model_selection.py
from __future__ import annotations
import io, os, time, json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drst_common.config import MODEL_DIR, RESULT_DIR, ACC_THR
from drst_common.minio_helper import save_bytes
from drst_common.resource_probe import start as start_probe
from drst_common.metric_logger import log_metric

from drst_forecasting.dataset import build_sliding_window
from drst_forecasting.models import LSTMForecaster
from drst_forecasting.registry import save_registry

# --------- 通用指标 ---------
def _rel_acc(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.05) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float((np.abs(y_pred - y_true) / denom <= thr).mean())

def _horizon_acc(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.05) -> List[float]:
    return [ _rel_acc(y_true[:,i], y_pred[:,i], thr) for i in range(y_true.shape[1]) ]

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, per_sample_ms: float) -> Dict[str, Any]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2":  float(r2_score(y_true, y_pred)),
        "acc@5%": _rel_acc(y_true, y_pred, 0.05),
        "acc@t+k": _horizon_acc(y_true, y_pred, 0.05),
        "latency_ms": float(per_sample_ms),
    }

# --------- LSTM 训练与预测 ---------
def _train_lstm(Xtr, Ytr, Xva, Yva, in_dim: int, H: int,
                hidden: int, layers: int, dropout: float, lr: float, bs: int, epochs: int) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMForecaster(in_dim=in_dim, hidden=hidden, num_layers=layers, horizon=H, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=bs, shuffle=True, drop_last=False)

    best = None; best_rmse = 1e18
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device).float(); yb = yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            pred = model(xb); loss = lossf(pred, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((pv - Yva)**2)))
        if rmse < best_rmse: best_rmse, best = rmse, io.BytesIO(); torch.save(model.to("cpu"), best); model.to(device)
    model = torch.load(io.BytesIO(best.getvalue()), map_location=device)
    model.eval()
    return model

def _infer_torch(model: nn.Module, X: np.ndarray) -> Tuple[np.ndarray, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    t0 = time.time()
    with torch.no_grad():
        pv = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    per_ms = (time.time() - t0) / len(X) * 1000.0
    return pv, per_ms

# --------- TransformerLight（轻量 PyTorch 版）---------
class TransformerLight(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, layers: int = 2, heads: int = 4, horizon: int = 5, dropout: float = 0.1):
        super().__init__()
        self.prj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4*d_model,
                                               batch_first=True, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):     # x: [B,T,D]
        z = self.prj(x)       # [B,T,d_model]
        z = self.encoder(z)   # [B,T,d_model]
        # global average over T
        z = z.transpose(1,2)  # [B,d_model,T]
        z = self.pool(z).squeeze(-1)  # [B,d_model]
        out = self.head(z)    # [B,H]
        return out

def _train_transformer(Xtr, Ytr, Xva, Yva, in_dim: int, H: int, d_model: int, layers: int, heads: int,
                       lr: float, bs: int, epochs: int, dropout: float) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLight(in_dim, d_model=d_model, layers=layers, heads=heads, horizon=H, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=bs, shuffle=True, drop_last=False)

    best = None; best_rmse = 1e18
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device).float(); yb = yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            pred = model(xb); loss = lossf(pred, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((pv - Yva)**2)))
        if rmse < best_rmse: best_rmse, best = rmse, io.BytesIO(); torch.save(model.to("cpu"), best); model.to(device)
    model = torch.load(io.BytesIO(best.getvalue()), map_location=device)
    model.eval()
    return model

# --------- 主流程：多模型 + 小网格（按照老师文字） ---------
LOOKBACK  = int(os.getenv("MS_LOOKBACK",  "10"))  # 老师文中 10
HORIZON   = int(os.getenv("MS_HORIZON",   "5"))   # 老师文中 5
EPOCHS    = int(os.getenv("MS_EPOCHS",    "80"))  # 可被环境覆盖
BS        = int(os.getenv("MS_BS",        "64"))

def main():
    stop = start_probe("model_selection")
    print(f"[model_selection] lookback={LOOKBACK} horizon={HORIZON}", flush=True)

    X, Y, feats = build_sliding_window(LOOKBACK, HORIZON)          # X:[N,T,D], Y:[N,H]
    N = len(X); n_va = max(1, int(0.3*N))
    Xtr, Xva = X[:-n_va], X[-n_va:]
    Ytr, Yva = Y[:-n_va], Y[-n_va:]

    results = []

    # -------- Ridge（多输出）--------
    ridge_grids = [0.01, 0.1, 1.0, 10.0]
    Xtr_flat, Xva_flat = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    best_ridge = None; best_r = None
    for a in ridge_grids:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", MultiOutputRegressor(Ridge(alpha=a)))])
        t0 = time.time(); pipe.fit(Xtr_flat, Ytr); pv = pipe.predict(Xva_flat); per_ms = (time.time()-t0)/len(Xva_flat)*1000
        evals = _evaluate(Yva, pv, per_ms); evals["model"]="ridge"; evals["alpha"]=a
        results.append(evals)
        if (best_r is None) or (evals["mae"] < best_r["mae"]): best_r, best_ridge = evals, pipe
    # dump ridge if best
    if best_ridge is not None:
        bio = io.BytesIO(); import joblib; joblib.dump(best_ridge, bio)
        save_bytes(f"{MODEL_DIR}/forecasting/ridge.pkl", bio.getvalue(), "application/octet-stream")

    # -------- XGBoost（可选）--------
    if HAS_XGB:
        xgb_params = [(50,3), (100,3), (200,4)]
        best_xgb=None; best_x=None
        for n, depth in xgb_params:
            model = MultiOutputRegressor(XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=0.1,
                                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                      tree_method="hist"))
            t0 = time.time(); model.fit(Xtr_flat, Ytr); pv = model.predict(Xva_flat)
            per_ms = (time.time()-t0)/len(Xva_flat)*1000
            evals = _evaluate(Yva, pv, per_ms); evals["model"]="xgboost"; evals["n"]=n; evals["max_depth"]=depth
            results.append(evals)
            if (best_x is None) or (evals["mae"] < best_x["mae"]): best_x, best_xgb = evals, model
        if best_xgb is not None:
            bio = io.BytesIO(); import joblib; joblib.dump(best_xgb, bio)
            save_bytes(f"{MODEL_DIR}/forecasting/xgboost.pkl", bio.getvalue(), "application/octet-stream")

    # -------- LSTM（标准多步）--------
    lstm_grid = [
        {"hidden":64, "layers":1, "dropout":0.1, "lr":1e-3},
        {"hidden":128,"layers":1, "dropout":0.3, "lr":5e-4},
    ]
    best_t = None; best_tm = None
    for g in lstm_grid:
        m = _train_lstm(Xtr, Ytr, Xva, Yva, X.shape[-1], HORIZON,
                        g["hidden"], g["layers"], g["dropout"],
                        g["lr"], BS, EPOCHS)
        pv, per_ms = _infer_torch(m, Xva)
        evals = _evaluate(Yva, pv, per_ms); evals["model"]="lstm"; evals.update(g); results.append(evals)
        if (best_t is None) or (evals["mae"] < best_t["mae"]): best_t, best_tm = evals, m
    if best_tm is not None:
        bio = io.BytesIO(); torch.save(best_tm.to("cpu"), bio)
        save_bytes(f"{MODEL_DIR}/forecasting/lstm.pt", bio.getvalue(), "application/octet-stream")

    # -------- TransformerLight --------
    tr_grid = [
        {"d_model":128, "layers":2, "heads":4, "dropout":0.1, "lr":1e-3},
        {"d_model":256, "layers":4, "heads":8, "dropout":0.1, "lr":1e-4},
    ]
    best_tr=None; best_trm=None
    for g in tr_grid:
        m = _train_transformer(Xtr, Ytr, Xva, Yva, X.shape[-1], HORIZON,
                               g["d_model"], g["layers"], g["heads"],
                               g["lr"], BS, EPOCHS, g["dropout"])
        pv, per_ms = _infer_torch(m, Xva)
        evals = _evaluate(Yva, pv, per_ms); evals["model"]="transformer_light"; evals.update(g); results.append(evals)
        if (best_tr is None) or (evals["mae"] < best_tr["mae"]): best_tr, best_trm = evals, m
    if best_trm is not None:
        bio = io.BytesIO(); torch.save(best_trm.to("cpu"), bio)
        save_bytes(f"{MODEL_DIR}/forecasting/transformer.pt", bio.getvalue(), "application/octet-stream")

    # -------- 选择最优（以 MAE 为主，可换成综合评分）--------
    df = pd.DataFrame(results).sort_values("mae")
    csv = df.to_csv(index=False).encode("utf-8")
    save_bytes(f"{RESULT_DIR}/forecasting/model_rank.csv", csv, "text/csv")

    best_row = df.iloc[0].to_dict()
    best_type = best_row["model"]
    if best_type == "ridge":
        artifact = f"{MODEL_DIR}/forecasting/ridge.pkl"
        framework = "sklearn"
    elif best_type == "xgboost":
        artifact = f"{MODEL_DIR}/forecasting/xgboost.pkl"
        framework = "sklearn"
    elif best_type == "lstm":
        artifact = f"{MODEL_DIR}/forecasting/lstm.pt"
        framework = "pytorch"
    else: # transformer_light
        artifact = f"{MODEL_DIR}/forecasting/transformer.pt"
        framework = "pytorch"

    registry = {
        "task": "throughput_forecasting",
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "best_model": best_type,
        "framework": framework,
        "artifact_key": artifact,
        "metrics": {k:best_row[k] for k in ["mae","r2","acc@5%","latency_ms"]},
        "note": "auto-selected by analysis/model_selection.py",
    }
    save_registry(registry)

    log_metric(component="forecasting", event="model_selection_done",
               mae=float(best_row["mae"]), r2=float(best_row["r2"]),
               acc=float(best_row["acc@5%"]), latency_ms=float(best_row["latency_ms"]))
    print("[model_selection] done. winner =", best_type, flush=True)
    stop()

if __name__ == "__main__":
    main()
