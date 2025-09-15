# drst_model_selection/pcm_select.py
from __future__ import annotations
import io, json
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost 可选
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# 轻量 Transformer（单输出）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drst_common.minio_helper import save_bytes
from drst_common.config import MODEL_DIR
from drst_forecasting.dataset import build_sliding_window
from .common import evaluate, bench_latency, save_rank_csv

# ---- 简易单输出 Transformer ----
class TransformerLight1D(nn.Module):
    def __init__(self, in_feat: int, d_model: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True, dropout=dropout, dim_feedforward=4 * d_model, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, L, F)
        z = self.proj(x)
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)

def _fit_torch_singleout(net: nn.Module, X: np.ndarray, y: np.ndarray, batch: int = 64, lr: float = 1e-3, epochs: int = 80, patience: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float().view(-1))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()

    best = None
    best_val = float("inf")
    bad = 0
    # 简单留出末尾 10% 作为 val
    n = len(X); v = max(1, int(0.1 * n))
    Xtr, ytr = X[:-v], y[:-v]
    Xva, yva = X[-v:], y[-v:]

    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float().view(-1)), batch_size=batch, shuffle=True):
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = lossf(pred, yb)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
        val = float(np.mean(np.abs(pv - yva)))
        if val + 1e-9 < best_val:
            best_val = val; bad = 0
            buf = io.BytesIO(); torch.save(net.to("cpu"), buf); net.to(device)
            best = buf.getvalue()
        else:
            bad += 1
            if bad > patience:
                break
    if best is not None:
        net = torch.load(io.BytesIO(best), map_location=device)
    net.eval()
    return net

def run_pcm_selection(lookback: int = 10, horizon: int = 5, take_last: int = 4000, topk: int = 3) -> None:
    """
    目标：单输出（预测 t+H 的值），和你的数据构造保持一致。
    候选：Ridge / RandomForest / GradientBoosting / XGBoost / TransformerLight(单输出)
    """
    print(f"[pcm_select] lookback={lookback} horizon(H)={horizon} take_last={take_last}", flush=True)

    # X: (N, L, F), y: (N,) —— build_sliding_window 只返回单值 y_{t+H}
    X, y, feats = build_sliding_window(lookback, horizon, take_last_n=take_last)
    N = len(X); assert N > 32, "not enough samples"
    v = max(1, int(0.3 * N))
    Xtr, Xva = X[:-v], X[-v:]
    ytr, yva = y[:-v], y[-v:]

    rows: List[Dict[str, Any]] = []

    # ---------- 传统模型（单输出） ----------
    # Ridge
    for a in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        mdl = Pipeline([("sc", StandardScaler(with_mean=False)), ("rg", Ridge(alpha=a))])
        Xtr_f = Xtr.reshape(len(Xtr), -1); Xva_f = Xva.reshape(len(Xva), -1)
        mdl.fit(Xtr_f, ytr)
        yp = mdl.predict(Xva_f)
        lat = bench_latency(lambda Z: mdl.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "Ridge", "alpha": a})
        rows.append(ev)

    # RandomForest
    for n in [200, 400]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=0)
        Xtr_f = Xtr.reshape(len(Xtr), -1); Xva_f = Xva.reshape(len(Xva), -1)
        rf.fit(Xtr_f, ytr)
        yp = rf.predict(Xva_f)
        lat = bench_latency(lambda Z: rf.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "RandomForest", "n_estimators": n})
        rows.append(ev)

    # GradientBoosting
    for n in [200, 400]:
        gb = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, max_depth=3, random_state=0)
        Xtr_f = Xtr.reshape(len(Xtr), -1); Xva_f = Xva.reshape(len(Xva), -1)
        gb.fit(Xtr_f, ytr)
        yp = gb.predict(Xva_f)
        lat = bench_latency(lambda Z: gb.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "GradientBoosting", "n_estimators": n})
        rows.append(ev)

    # XGBoost（可选）
    if _HAS_XGB:
        for n in [200, 400]:
            xgb = XGBRegressor(
                n_estimators=n, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0, objective="reg:squarederror", tree_method="hist", n_jobs=0
            )
            Xtr_f = Xtr.reshape(len(Xtr), -1); Xva_f = Xva.reshape(len(Xva), -1)
            xgb.fit(Xtr_f, ytr)
            yp = xgb.predict(Xva_f)
            lat = bench_latency(lambda Z: xgb.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
            ev = evaluate(yva, yp, lat); ev.update({"model": "XGBoost", "n_estimators": n})
            rows.append(ev)

    # ---------- 轻量 Transformer（单输出） ----------
    try:
        net = TransformerLight1D(in_feat=X.shape[-1], d_model=128, heads=4, layers=2, dropout=0.1)
        net = _fit_torch_singleout(net, Xtr, ytr, batch=64, lr=1e-3, epochs=80, patience=10)
        # 预测 + 延迟
        def _torch_pred(Z: np.ndarray) -> np.ndarray:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net.eval()
            out = []
            with torch.no_grad():
                for xb in DataLoader(torch.from_numpy(Z).float(), batch_size=128):
                    out.append(net(xb.to(device)).cpu().numpy())
            return np.concatenate(out, axis=0)
        yp = _torch_pred(Xva)
        lat = bench_latency(_torch_pred, Xva, repeat=1)
        ev = evaluate(yva, yp, lat); ev.update({"model": "TransformerLight1D"})
        rows.append(ev)
    except Exception as e:
        print(f"[pcm_select] transformer skipped: {e}", flush=True)

    # ---------- 排序/保存 ----------
    df = pd.DataFrame(rows).sort_values(["mae", "latency_ms"]).reset_index(drop=True)
    csv_key = save_rank_csv("pcm_model_selection.csv", df)
    print(f"[pcm_select] wrote rank -> s3://.../{csv_key}", flush=True)

    # 只写一个轻量“建议”清单，供后续 GridSearch 参考（不覆盖 selected.json）
    best = df.iloc[0].to_dict()
    suggestion = {
        "task": "pcm",
        "lookback": lookback,
        "horizon": horizon,
        "features": feats,
        "recommend_topk": json.loads(df.head(min(len(df), max(1, int(topk)))).to_json(orient="records")),
        "winner": best.get("model"),
        "note": "Single-output selection (predict y_{t+H}); grid-search can down-select using 'recommend_topk'."
    }
    save_bytes(f"{MODEL_DIR}/forecasting/pcm_select_suggestion.json",
               json.dumps(suggestion, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
    print(f"[pcm_select] winner={best.get('model')} mae={best.get('mae'):.4f}", flush=True)
