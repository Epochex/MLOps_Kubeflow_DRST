# drst_model_selection/pcm_select.py
from __future__ import annotations
import io, json, time
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


# ---------- 小工具 ----------
def _seed_everything(seed: int = 42):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass


def _canon_name(name: str) -> str | None:
    """把 selection 模型名映射到 train_benchmark 认识的名字"""
    n = str(name).strip().lower()
    if n in ("ridge",):
        return "ridge"
    if n in ("xgboost", "xgb", "xgbregressor"):
        return "xgboost"
    if n in ("transformerlight1d", "transformerlight", "transformer"):
        return "transformerlight"
    if n in ("randomforest", "random_forest", "rf", "gradientboosting", "gbrt", "gbdt"):
        # 训练阶段没有 RF/GBDT 的实现，用 XGB 近似替代
        return "xgboost"
    return None


# ---- 简易单输出 Transformer ----
class TransformerLight1D(nn.Module):
    def __init__(self, in_feat: int, d_model: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True, dropout=dropout,
            dim_feedforward=4 * d_model, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, L, F)
        z = self.proj(x)
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)


def _fit_torch_singleout(
    net: nn.Module, X: np.ndarray, y: np.ndarray,
    batch: int = 64, lr: float = 1e-3, epochs: int = 80, patience: int = 10
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    lossf = nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # 简单留出末尾 10% 作为 val
    n = len(X)
    v = max(1, int(0.1 * n))
    Xtr, ytr = X[:-v], y[:-v]
    Xva, yva = X[-v:], y[-v:]

    def _epoch(dl_X: np.ndarray, dl_y: np.ndarray, train: bool) -> float:
        total, cnt = 0.0, 0
        if train:
            net.train()
        else:
            net.eval()
        for i in range(0, len(dl_X), batch):
            xb = torch.from_numpy(dl_X[i:i+batch]).float().to(device)
            yb = torch.from_numpy(dl_y[i:i+batch]).float().to(device)
            if train:
                opt.zero_grad(set_to_none=True)
                pred = net(xb)
                loss = lossf(pred, yb)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    pred = net(xb)
                    loss = lossf(pred, yb)
            total += float(loss.detach().cpu().item()) * len(xb)
            cnt += len(xb)
        return total / max(1, cnt)

    best_blob, best_val, bad = None, float("inf"), 0
    for ep in range(1, epochs + 1):
        tr = _epoch(Xtr, ytr, True)
        va = _epoch(Xva, yva, False)
        print(f"[pcm_select][Transformer] epoch={ep:03d} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)
        if va + 1e-9 < best_val:
            best_val, bad = va, 0
            buf = io.BytesIO()
            torch.save(net.to("cpu"), buf)
            net.to(device)
            best_blob = buf.getvalue()
        else:
            bad += 1
            if bad > patience:
                print(f"[pcm_select][Transformer] early stop at epoch {ep} (patience={patience})", flush=True)
                break

    if best_blob is not None:
        net = torch.load(io.BytesIO(best_blob), map_location=device)
    net.eval()
    return net


# ---------- 主流程 ----------
def run_pcm_selection(lookback: int = 10, horizon: int = 5, take_last: int = 4000, topk: int = 3) -> None:
    """
    单输出选择（预测 y_{t+H}）。
    候选：Ridge / RandomForest / GradientBoosting / XGBoost / TransformerLight(单输出)
    """
    _seed_everything(42)

    print(f"[pcm_select] === START ===", flush=True)
    print(f"[pcm_select] params: lookback={lookback} horizon(H)={horizon} take_last={take_last} topk={topk}", flush=True)

    t0 = time.perf_counter()
    # X: (N, L, F), y: (N,) —— build_sliding_window 默认即单步 y_{t+H}
    X, y, feats = build_sliding_window(lookback, horizon, take_last_n=take_last)
    print(f"[pcm_select] data: X.shape={X.shape} y.shape={y.shape} | #features={len(feats)}", flush=True)
    N = len(X)
    assert N > 32, "not enough samples"
    v = max(1, int(0.3 * N))
    Xtr, Xva = X[:-v], X[-v:]
    ytr, yva = y[:-v], y[-v:]
    print(f"[pcm_select] split: train={len(Xtr)} valid={len(Xva)} (val_frac≈{len(Xva)/N:.2%})", flush=True)

    rows: List[Dict[str, Any]] = []
    best_key = None  # (mae, latency_ms)

    # ---------- 传统模型（单输出） ----------
    # Ridge
    for a in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        print(f"[pcm_select][Ridge] alpha={a} ...", flush=True)
        mdl = Pipeline([("sc", StandardScaler(with_mean=False)), ("rg", Ridge(alpha=a))])
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xva_f = Xva.reshape(len(Xva), -1)
        t1 = time.perf_counter()
        mdl.fit(Xtr_f, ytr)
        fit_s = time.perf_counter() - t1
        yp = mdl.predict(Xva_f)
        lat = bench_latency(lambda Z: mdl.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat)
        ev.update({"model": "Ridge", "alpha": a, "fit_seconds": round(fit_s, 6)})
        rows.append(ev)
        print(f"[pcm_select][Ridge] r2={ev['r2']:.4f} mae={ev['mae']:.4f} lat_ms={ev['latency_ms']:.3f} fit_s={fit_s:.2f}", flush=True)

    # RandomForest
    for n in [200, 400]:
        print(f"[pcm_select][RandomForest] n_estimators={n} ...", flush=True)
        rf = RandomForestRegressor(n_estimators=n, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=0)
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xva_f = Xva.reshape(len(Xva), -1)
        t1 = time.perf_counter()
        rf.fit(Xtr_f, ytr)
        fit_s = time.perf_counter() - t1
        yp = rf.predict(Xva_f)
        lat = bench_latency(lambda Z: rf.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat)
        ev.update({"model": "RandomForest", "n_estimators": n, "fit_seconds": round(fit_s, 6)})
        rows.append(ev)
        print(f"[pcm_select][RandomForest] r2={ev['r2']:.4f} mae={ev['mae']:.4f} lat_ms={ev['latency_ms']:.3f} fit_s={fit_s:.2f}", flush=True)

    # GradientBoosting
    for n in [200, 400]:
        print(f"[pcm_select][GBDT] n_estimators={n} ...", flush=True)
        gb = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, max_depth=3, random_state=0)
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xva_f = Xva.reshape(len(Xva), -1)
        t1 = time.perf_counter()
        gb.fit(Xtr_f, ytr)
        fit_s = time.perf_counter() - t1
        yp = gb.predict(Xva_f)
        lat = bench_latency(lambda Z: gb.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
        ev = evaluate(yva, yp, lat)
        ev.update({"model": "GradientBoosting", "n_estimators": n, "fit_seconds": round(fit_s, 6)})
        rows.append(ev)
        print(f"[pcm_select][GBDT] r2={ev['r2']:.4f} mae={ev['mae']:.4f} lat_ms={ev['latency_ms']:.3f} fit_s={fit_s:.2f}", flush=True)

    # XGBoost（可选）
    if _HAS_XGB:
        for n in [200, 400]:
            print(f"[pcm_select][XGBoost] n_estimators={n} ...", flush=True)
            xgb = XGBRegressor(
                n_estimators=n, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0, objective="reg:squarederror", tree_method="hist", n_jobs=0
            )
            Xtr_f = Xtr.reshape(len(Xtr), -1)
            Xva_f = Xva.reshape(len(Xva), -1)
            t1 = time.perf_counter()
            xgb.fit(Xtr_f, ytr)
            fit_s = time.perf_counter() - t1
            yp = xgb.predict(Xva_f)
            lat = bench_latency(lambda Z: xgb.predict(Z.reshape(len(Z), -1)), Xva, repeat=1)
            ev = evaluate(yva, yp, lat)
            ev.update({"model": "XGBoost", "n_estimators": n, "fit_seconds": round(fit_s, 6)})
            rows.append(ev)
            print(f"[pcm_select][XGBoost] r2={ev['r2']:.4f} mae={ev['mae']:.4f} lat_ms={ev['latency_ms']:.3f} fit_s={fit_s:.2f}", flush=True)
    else:
        print("[pcm_select][XGBoost] package not available in image — skipped.", flush=True)

    # ---------- 轻量 Transformer（单输出） ----------
    try:
        print("[pcm_select][Transformer] training ...", flush=True)
        net = TransformerLight1D(in_feat=X.shape[-1], d_model=128, heads=4, layers=2, dropout=0.1)
        t1 = time.perf_counter()
        net = _fit_torch_singleout(net, Xtr, ytr, batch=64, lr=1e-3, epochs=80, patience=10)
        fit_s = time.perf_counter() - t1

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
        ev = evaluate(yva, yp, lat)
        ev.update({"model": "TransformerLight1D", "fit_seconds": round(fit_s, 6)})
        rows.append(ev)
        print(f"[pcm_select][Transformer] r2={ev['r2']:.4f} mae={ev['mae']:.4f} lat_ms={ev['latency_ms']:.3f} fit_s={fit_s:.2f}", flush=True)
    except Exception as e:
        print(f"[pcm_select][Transformer] skipped due to error: {e}", flush=True)

    # ---------- 排序/保存 ----------
    df = pd.DataFrame(rows).sort_values(["mae", "latency_ms"]).reset_index(drop=True)
    csv_key = save_rank_csv("pcm_model_selection.csv", df)
    print(f"[pcm_select] wrote rank CSV -> models in s3://.../{csv_key}", flush=True)
    print(f"[pcm_select] top-5 preview:\n{df.head(5).to_string(index=False)}", flush=True)

    best = df.iloc[0].to_dict()
    print(f"[pcm_select] WINNER model={best.get('model')} mae={best.get('mae'):.6f} r2={best.get('r2'):.6f} lat_ms={best.get('latency_ms'):.3f}", flush=True)

    # 生成供 gridsearch 使用的候选（去重保序，裁剪 topk）
    ordered = [str(m) for m in df["model"].tolist()]
    mapped = []
    for m in ordered:
        c = _canon_name(m)
        if c and c not in mapped:
            mapped.append(c)
    if not mapped:
        mapped = ["xgboost", "transformerlight", "ridge"]
    candidates = mapped[:max(1, int(topk))]

    payload = {"task": "pcm", "topk": int(topk), "candidates": candidates}
    key = f"{MODEL_DIR}/forecast/model_candidates.json"
    save_bytes(key, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
    print(f"[pcm_select] published candidates -> s3://.../{key} :: {json.dumps(payload, ensure_ascii=False)}", flush=True)

    dt = time.perf_counter() - t0
    print(f"[pcm_select] === DONE in {dt:.2f}s ===", flush=True)
