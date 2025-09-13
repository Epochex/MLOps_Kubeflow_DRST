#!/usr/bin/env python3
# drst_forecasting/models.py
from __future__ import annotations
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import joblib

from drst_common.minio_helper import save_bytes, s3, BUCKET
from drst_common.config import MODEL_DIR

# 传统模型
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# XGBoost（训练镜像需要装 xgboost，见下方 Dockerfile）
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# 深度学习
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 工具
# ======================
def now_ts() -> int:
    return int(time.time())

def to_bytes_joblib(obj) -> bytes:
    bio = io.BytesIO()
    joblib.dump(obj, bio)
    return bio.getvalue()

def from_bytes_joblib(b: bytes):
    bio = io.BytesIO(b)
    return joblib.load(bio)

def to_bytes_torch(model: nn.Module) -> bytes:
    bio = io.BytesIO()
    torch.save(model, bio)
    return bio.getvalue()

def from_bytes_torch(b: bytes) -> nn.Module:
    bio = io.BytesIO(b)
    return torch.load(bio, map_location="cpu")

def _flatten_if_seq(X: np.ndarray) -> np.ndarray:
    # 允许输入为 (N, lookback, F) 或 (N, lookback*F)
    if X.ndim == 3:
        N, L, F = X.shape
        return X.reshape(N, L * F)
    return X

def _ensure_2d_y(Y: np.ndarray) -> np.ndarray:
    # 输出统一成 (N, H)
    if Y.ndim == 1:
        return Y.reshape(-1, 1)
    return Y

# ======================
# 传统模型：Ridge / XGB
# ======================
@dataclass
class SkModelCfg:
    name: str
    params: Dict[str, Any]
    lookback: int
    horizon: int
    features: List[str]

class SkWrapper:
    def __init__(self, base, lookback:int, horizon:int, features:List[str], params:Dict[str,Any]):
        # horizon>1 用 MultiOutput 包一层
        if horizon > 1:
            self.model = MultiOutputRegressor(base.__class__(**params))
        else:
            self.model = base.__class__(**params)
        self.cfg = SkModelCfg(
            name=base.__class__.__name__,
            params=params,
            lookback=lookback,
            horizon=horizon,
            features=features
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X2 = _flatten_if_seq(X)
        Y2 = _ensure_2d_y(Y)
        self.model.fit(X2, Y2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2 = _flatten_if_seq(X)
        pred = self.model.predict(X2)
        return _ensure_2d_y(np.asarray(pred))

    def save_to_s3(self, key_prefix: str) -> Dict[str, Any]:
        model_key = f"{key_prefix}/best_model.joblib"
        meta_key  = f"{key_prefix}/best_meta.json"
        save_bytes(model_key, to_bytes_joblib(self.model), "application/octet-stream")
        meta = asdict(self.cfg)
        meta["artifact"] = model_key
        save_bytes(meta_key, json.dumps(meta, ensure_ascii=False, indent=2).encode(), "application/json")
        return {"model_key": model_key, "meta_key": meta_key, "meta": meta}

    @staticmethod
    def load_from_s3(meta_key: str) -> "SkWrapper":
        raw = s3.get_object(Bucket=BUCKET, Key=meta_key)["Body"].read()
        meta = json.loads(raw.decode("utf-8"))
        model_raw = s3.get_object(Bucket=BUCKET, Key=meta["artifact"])["Body"].read()
        obj = SkWrapper(Ridge(), lookback=meta["lookback"], horizon=meta["horizon"], features=meta["features"], params={})
        obj.model = from_bytes_joblib(model_raw)
        obj.cfg = SkModelCfg(**{k: meta[k] for k in ("name","params","lookback","horizon","features")})
        return obj

# ======================
# 深度模型：LSTM / DirectLSTM / TransformerLight
# ======================
class LSTMHead(nn.Module):
    def __init__(self, in_feat:int, hidden:int, layers:int, dropout:float, horizon:int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=hidden, num_layers=layers, dropout=(dropout if layers>1 else 0.0), batch_first=True)
        self.out  = nn.Linear(hidden, horizon)
    def forward(self, x):  # x: (B, L, F)
        y, _ = self.lstm(x)
        last = y[:, -1, :]
        return self.out(last)

class TransformerLight(nn.Module):
    def __init__(self, in_feat:int, d_model:int, heads:int, dropout:float, horizon:int):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
        self.norm1= nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, horizon)
    def forward(self, x):  # (B,L,F)
        z = self.proj(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        f = self.ffn(z)
        z = self.norm2(z + f)
        last = z[:, -1, :]
        return self.out(last)

@dataclass
class TorchCfg:
    name: str
    params: Dict[str, Any]
    lookback: int
    horizon: int
    features: List[str]

class TorchWrapper:
    def __init__(self, net: nn.Module, lookback:int, horizon:int, features:List[str], params:Dict[str,Any], lr:float=1e-3, batch:int=64, max_epoch:int=200, patience:int=10):
        self.net = net
        self.cfg = TorchCfg(name=net.__class__.__name__, params=params, lookback=lookback, horizon=horizon, features=features)
        self.lr = float(lr); self.batch=int(batch); self.max_epoch=int(max_epoch); self.patience=int(patience)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 3, "Torch models expect X as (N, lookback, F)"
        Y2 = _ensure_2d_y(Y)
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y2).float())
        dl = DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.SmoothL1Loss()
        best = None; bad=0; best_val=float("inf")
        # 简单留出法：用最后 10% 当 val
        n = len(ds); val_n=max(1,int(n*0.1))
        Xtr, Ytr = X[:-val_n], Y2[:-val_n]
        Xva, Yva = X[-val_n:], Y2[-val_n:]
        for ep in range(1, self.max_epoch+1):
            self.net.train()
            for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()), batch_size=self.batch, shuffle=True):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = self.net(xb)
                loss = lossf(pred, yb)
                loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                pv = []
                for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float()), batch_size=self.batch):
                    xb = xb.to(self.device)
                    pv.append(self.net(xb).cpu().numpy())
                pv = np.concatenate(pv, axis=0)
                val = float(np.mean(np.abs(pv - Yva)))
            if val + 1e-9 < best_val:
                best_val = val; bad = 0
                best = to_bytes_torch(self.net.cpu()); self.net.to(self.device)
            else:
                bad += 1
                if bad > self.patience:
                    break
        if best is not None:
            self.net = from_bytes_torch(best).to(self.device)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3
        self.net.eval()
        out = []
        with torch.no_grad():
            for xb in DataLoader(torch.from_numpy(X).float(), batch_size=self.batch):
                xb = xb.to(self.device)
                out.append(self.net(xb).cpu().numpy())
        return _ensure_2d_y(np.concatenate(out, axis=0))

    def save_to_s3(self, key_prefix: str) -> Dict[str, Any]:
        model_key = f"{key_prefix}/best_model.pt"
        meta_key  = f"{key_prefix}/best_meta.json"
        save_bytes(model_key, to_bytes_torch(self.net.cpu()), "application/octet-stream")
        meta = asdict(self.cfg); meta["artifact"] = model_key
        save_bytes(meta_key, json.dumps(meta, ensure_ascii=False, indent=2).encode(), "application/json")
        return {"model_key": model_key, "meta_key": meta_key, "meta": meta}

    @staticmethod
    def load_from_s3(meta_key: str) -> "TorchWrapper":
        raw = s3.get_object(Bucket=BUCKET, Key=meta_key)["Body"].read()
        meta = json.loads(raw.decode("utf-8"))
        model_raw = s3.get_object(Bucket=BUCKET, Key=meta["artifact"])["Body"].read()
        net = from_bytes_torch(model_raw)
        obj = TorchWrapper(net, lookback=meta["lookback"], horizon=meta["horizon"], features=meta["features"], params=meta["params"])
        obj.net = net.to(obj.device)
        return obj

# ======================
# 工厂 & 统一入口
# ======================
def build_model(kind:str, lookback:int, horizon:int, features:List[str], params:Dict[str,Any]) -> Union[SkWrapper, TorchWrapper]:
    k = kind.lower()
    if k == "ridge":
        return SkWrapper(Ridge(alpha=float(params.get("alpha", 1.0))), lookback, horizon, features, params)
    if k == "xgboost":
        if not _HAS_XGB:
            raise RuntimeError("xgboost 未安装，训练镜像需要包含 xgboost 包")
        base = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            tree_method="hist",
            n_jobs=0
        )
        return SkWrapper(base, lookback, horizon, features, params)
    if k == "lstm":
        hidden=int(params.get("hidden",64)); layers=int(params.get("layers",1)); dropout=float(params.get("dropout",0.1))
        lr=float(params.get("lr",1e-3)); batch=int(params.get("batch",64)); max_epoch=int(params.get("epochs",200)); patience=int(params.get("patience",10))
        net=LSTMHead(in_feat=len(features), hidden=hidden, layers=layers, dropout=dropout, horizon=horizon)
        return TorchWrapper(net, lookback, horizon, features, params, lr=lr, batch=batch, max_epoch=max_epoch, patience=patience)
    if k == "directlstm":
        hidden=int(params.get("hidden",128)); layers=int(params.get("layers",2)); dropout=float(params.get("dropout",0.1))
        lr=float(params.get("lr",5e-4)); batch=int(params.get("batch",64)); max_epoch=int(params.get("epochs",200)); patience=int(params.get("patience",10))
        net=LSTMHead(in_feat=len(features), hidden=hidden, layers=layers, dropout=dropout, horizon=horizon)
        return TorchWrapper(net, lookback, horizon, features, params, lr=lr, batch=batch, max_epoch=max_epoch, patience=patience)
    if k == "transformerlight":
        d_model=int(params.get("d_model",128)); heads=int(params.get("heads",4)); dropout=float(params.get("dropout",0.1))
        lr=float(params.get("lr",1e-3)); batch=int(params.get("batch",64)); max_epoch=int(params.get("epochs",200)); patience=int(params.get("patience",10))
        net=TransformerLight(in_feat=len(features), d_model=d_model, heads=heads, dropout=dropout, horizon=horizon)
        return TorchWrapper(net, lookback, horizon, features, params, lr=lr, batch=batch, max_epoch=max_epoch, patience=patience)
    raise ValueError(f"unknown model kind: {kind}")

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # 统一用第一步（t+1）评价；如果是多步输出，取第 0 列
    y_true = _ensure_2d_y(y_true); y_pred = _ensure_2d_y(y_pred)
    if y_true.shape[1] > 1:
        yt = y_true[:, 0]
        yp = y_pred[:, 0]
    else:
        yt = y_true.ravel()
        yp = y_pred.ravel()
    r2 = float(r2_score(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    # 短期可靠性 Acc@0.15：相对误差 <= 0.15
    denom = np.maximum(np.abs(yt), 1e-8)
    acc = float(np.mean(np.abs(yp - yt) / denom <= 0.15))
    return {"r2": r2, "mae": mae, "acc@0.15": acc}

def benchmark_latency(predict_fn, X: np.ndarray, repeat:int=1) -> float:
    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        _ = predict_fn(X)
    dt = (time.perf_counter() - t0) / max(1, repeat)
    return float(dt / len(X))  # per-sample seconds

def publish_best_selection(prefix:str, info:Dict[str,Any]) -> None:
    """
    在 S3 写入： models/forecast/best_meta.json + models/forecast/selected.json
    API 会读取 selected.json 定位 meta，再取模型。
    """
    sel_key = f"{prefix}/selected.json"
    save_bytes(sel_key, json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
