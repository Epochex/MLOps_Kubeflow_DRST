# drst_model_selection/pcm.py
from __future__ import annotations
import io, json, time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drst_common.config import RESULT_DIR, MODEL_DIR
from drst_common.minio_helper import save_bytes
from drst_forecasting.dataset import build_sliding_window
from drst_forecasting.models import TorchWrapper, TransformerLight as TFMLight

def _ensure_2d(y):
    y = np.asarray(y, dtype=float)
    return y if y.ndim == 2 else y.reshape(-1, 1)

def _rel_acc(y_true, y_pred, thr=0.05):
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), 1e-8)
    return float(np.mean(np.abs(yp - yt) / denom <= float(thr)))

def _horizon_acc(y_true, y_pred, thr=0.05) -> List[float]:
    yt = _ensure_2d(y_true); yp = _ensure_2d(y_pred)
    return [ _rel_acc(yt[:,i], yp[:,i], thr) for i in range(yt.shape[1]) ]

def _evaluate(y_true, y_pred, lat_ms: float) -> Dict[str, Any]:
    yt = _ensure_2d(y_true); yp = _ensure_2d(y_pred)
    mae = float(np.mean(np.abs(yp - yt)))
    # 简化 r2：对第一步
    from sklearn.metrics import r2_score
    r2 = float(r2_score(yt[:,0], yp[:,0]))
    return {"mae":mae, "r2":r2, "acc@5%":_rel_acc(yt, yp), "acc@t+k":_horizon_acc(yt, yp), "latency_ms":float(lat_ms)}

def _bench_latency(fn, X):
    t0 = time.perf_counter(); _ = fn(X); dt = time.perf_counter() - t0
    return float(dt / len(X) * 1000.0)

# ---- 轻量 TCN（CPU 友好）----
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dil=1, dropout=0.1):
        super().__init__()
        pad = (k-1) * dil
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dil)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dil)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):     # (B,C,T)
        y = self.conv1(x); y = self.act(y); y = self.drop(y)
        y = self.conv2(y); y = self.act(y); y = self.drop(y)
        return y + self.res(x)

class TCNLight(nn.Module):
    def __init__(self, in_feat:int, horizon:int, ch:int=64, layers:int=3, dropout:float=0.1):
        super().__init__()
        blocks = []
        C = in_feat
        for i in range(layers):
            blocks.append(TCNBlock(C, ch, k=3, dil=2**i, dropout=dropout))
            C = ch
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(ch, horizon)
    def forward(self, x):                 # x: (B,L,F)
        z = x.transpose(1,2)              # -> (B,F,L)
        z = self.net(z)                   # -> (B,C,L)
        last = z[:, :, -1]                # -> (B,C)
        return self.head(last)            # -> (B,H)

def run_pcm_selection(lookback:int=10, horizon:int=5, take_last:int=4000, topk:int=3):
    # 1) 滑窗
    X, Y, feats = build_sliding_window(lookback, horizon, take_last_n=(None if int(take_last)<=0 else int(take_last)))
    N = len(X); v = max(1, int(0.3*N))
    Xtr, Xva = X[:-v], X[-v:]
    Ytr, Yva = Y[:-v], Y[-v:]

    rows = []

    # ---- 树模型（直接多步）：flatten + MultiOutput ----
    Xtr_f, Xva_f = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)

    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=0))
    rf.fit(Xtr_f, Ytr)
    lat = _bench_latency(lambda Z: rf.predict(Z.reshape(len(Z), -1)), Xva)
    rows.append({"model":"RandomForest", **_evaluate(Yva, rf.predict(Xva_f), lat)})

    gbr = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=200, random_state=0))
    gbr.fit(Xtr_f, Ytr)
    lat = _bench_latency(lambda Z: gbr.predict(Z.reshape(len(Z), -1)), Xva)
    rows.append({"model":"GradientBoosting", **_evaluate(Yva, gbr.predict(Xva_f), lat)})

    if _HAS_XGB:
        xgb = MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                                reg_lambda=1.0, random_state=0))
        xgb.fit(Xtr_f, Ytr)
        lat = _bench_latency(lambda Z: xgb.predict(Z.reshape(len(Z), -1)), Xva)
        rows.append({"model":"XGBoost", **_evaluate(Yva, xgb.predict(Xva_f), lat)})

    # ---- Transformer（轻量）----
    tfm = TorchWrapper(
        net=TFMLight(in_feat=X.shape[-1], d_model=128, heads=4, dropout=0.1, horizon=horizon),
        lookback=lookback, horizon=horizon, features=feats,
        params={"d_model":128,"heads":4,"dropout":0.1,"epochs":80},
        lr=1e-3, batch=64, max_epoch=80, patience=10
    ).fit(Xtr, Ytr)
    lat = _bench_latency(lambda Z: tfm.predict(Z), Xva)
    rows.append({"model":"TransformerLight", **_evaluate(Yva, tfm.predict(Xva), lat)})

    # ---- TCN（CPU 友好）----
    tcn = TorchWrapper(
        net=TCNLight(in_feat=X.shape[-1], horizon=horizon, ch=64, layers=3, dropout=0.1),
        lookback=lookback, horizon=horizon, features=feats,
        params={"channels":64,"layers":3,"dropout":0.1,"epochs":80},
        lr=1e-3, batch=64, max_epoch=80, patience=10
    ).fit(Xtr, Ytr)
    lat = _bench_latency(lambda Z: tcn.predict(Z), Xva)
    rows.append({"model":"TCNLight", **_evaluate(Yva, tcn.predict(Xva), lat)})

    # 2) 输出
    df = pd.DataFrame(rows).sort_values(["mae","latency_ms"], ascending=[True, True])
    save_bytes(f"{RESULT_DIR}/forecasting/pcm_model_rank.csv", df.to_csv(index=False).encode("utf-8"), "text/csv")

    top = df.head(max(1, int(topk))).to_dict(orient="records")
    save_bytes(f"{MODEL_DIR}/forecast/pcm_selection.json",
               json.dumps({"lookback":lookback,"horizon":horizon,"top":top}, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
    print("[pcm.model_selection] wrote:",
          f"s3://.../{RESULT_DIR}/forecasting/pcm_model_rank.csv and s3://.../{MODEL_DIR}/forecast/pcm_selection.json",
          flush=True)
