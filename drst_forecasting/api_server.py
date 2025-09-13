#!/usr/bin/env python3
# drst_forecasting/api_server.py
from __future__ import annotations
import json
from typing import Dict, List, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from drst_common.minio_helper import s3, BUCKET
from drst_common.config import MODEL_DIR
from .models import SkWrapper, TorchWrapper

app = FastAPI(title="DRST Forecast API", version="1.0.0")

class PredictIn(BaseModel):
    sequence: List[Dict[str, float]]

class PredictOut(BaseModel):
    horizon: int
    y_hat: List[float]

# 缓存
_state: Dict[str, Any] = {}

def _load_selection():
    sel_key = f"{MODEL_DIR}/forecast/selected.json"
    raw = s3.get_object(Bucket=BUCKET, Key=sel_key)["Body"].read()
    sel = json.loads(raw.decode("utf-8"))
    meta_raw = s3.get_object(Bucket=BUCKET, Key=sel["meta_key"])["Body"].read()
    meta = json.loads(meta_raw.decode("utf-8"))
    kind = str(sel["selected_kind"]).lower()
    if kind in ("ridge","xgboost"):
        wrapper = SkWrapper.load_from_s3(sel["meta_key"])
    else:
        wrapper = TorchWrapper.load_from_s3(sel["meta_key"])
    return {
        "sel": sel, "meta": meta,
        "wrapper": wrapper,
        "lookback": int(sel["lookback"]),
        "horizon": int(sel["horizon"]),
        "features": sel.get("features", meta.get("features", [])),
    }

@app.on_event("startup")
def _startup():
    global _state
    _state = _load_selection()

@app.get("/")
def health():
    return {"ok": True}

@app.get("/info")
def info():
    return {
        "selected": _state.get("sel", {}),
        "meta": _state.get("meta", {}),
        "features": _state.get("features", []),
        "lookback": _state.get("lookback"),
        "horizon": _state.get("horizon"),
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    seq = inp.sequence
    feats = _state["features"]; L = _state["lookback"]; H = _state["horizon"]
    if not isinstance(seq, list) or len(seq) < L:
        raise HTTPException(status_code=400, detail=f"sequence length must be >= lookback={L}")
    # 取最后 L 步（单条）
    tail = seq[-L:]
    x = np.zeros((1, L, len(feats)), dtype=np.float32)
    for t, row in enumerate(tail):
        for j, f in enumerate(feats):
            x[0, t, j] = float(row.get(f, 0.0))
    wrapper = _state["wrapper"]
    if hasattr(wrapper, "net"):  # torch
        y = wrapper.predict(x)[0]
    else:
        # 传统模型 expects 2D（flatten）
        y = wrapper.predict(x)[0]
    y = y.squeeze().tolist()
    if isinstance(y, float):
        y = [y]
    return PredictOut(horizon=H, y_hat=y[:H])
