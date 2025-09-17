# -*- coding: utf-8 -*-
"""
通用工件读写（MinIO）

功能：
- write_latest(model_bytes, metrics, model_key?, metrics_key?)  →  写入模型/指标，并更新 models/latest.json
- read_latest() → (model_key, metrics_key, ts)    兼容旧格式 latest.txt
- load_model_by_key(key) → (torch_model, raw_bytes)  相对路径会自动加 MODEL_DIR/
- load_scaler() / load_selected_feats()
"""

from __future__ import annotations
import io
import json
import time
from typing import Optional, Tuple, Dict, Any, List

import joblib
import numpy as np
import torch

from .minio_helper import s3, save_bytes
from .config import BUCKET, MODEL_DIR

# ------------- 基础 I/O -------------
def _abs_key(key: str) -> str:
    return key if ("/" in key) else f"{MODEL_DIR}/{key}"

def _read_bytes(key: str) -> Optional[bytes]:
    try:
        return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception:
        return None

def _read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        raw = _read_bytes(key)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _head_mtime(key: str) -> Optional[float]:
    try:
        return s3.head_object(Bucket=BUCKET, Key=key)["LastModified"].timestamp()
    except Exception:
        return None

# ------------- latest 指针 -------------
def write_latest(model_bytes: bytes,
                 metrics: Dict[str, Any],
                 model_key: Optional[str] = None,
                 metrics_key: Optional[str] = None) -> Tuple[str, str]:
    """
    把模型+指标写入 MODEL_DIR 下，并更新 latest.json
    返回：(model_key, metrics_key)
    """
    ts = int(time.time())
    model_key = model_key or f"model_{ts}.pt"
    metrics_key = metrics_key or f"metrics_{ts}.json"

    mkey_abs = _abs_key(model_key)
    with io.BytesIO(model_bytes) as bio:
        save_bytes(mkey_abs, bio.read(), "application/octet-stream")
    save_bytes(_abs_key(metrics_key), json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")

    latest = {"model_key": model_key, "metrics_key": metrics_key, "ts": ts}
    save_bytes(f"{MODEL_DIR}/latest.json", json.dumps(latest, ensure_ascii=False).encode("utf-8"), "application/json")
    # 兼容性：也写一个简易 txt（两行），老代码可用
    save_bytes(f"{MODEL_DIR}/latest.txt", f"{model_key}\n{metrics_key}\n".encode("utf-8"), "text/plain")
    return model_key, metrics_key

def read_latest() -> Optional[Tuple[str, str, int]]:
    """
    读取最新模型指针：
    - 优先 latest.json：{"model_key": "...", "metrics_key": "...", "ts": 1700000000}
    - 兼容 latest.txt：两行，第一行模型键、第二行指标键；ts 用对象的 mtime
    """
    js = _read_json(f"{MODEL_DIR}/latest.json")
    if js and "model_key" in js and "metrics_key" in js:
        ts = int(js.get("ts") or (_head_mtime(f"{MODEL_DIR}/{js['model_key']}") or 0))
        return str(js["model_key"]), str(js["metrics_key"]), ts

    # 兼容旧格式
    raw = _read_bytes(f"{MODEL_DIR}/latest.txt")
    if raw:
        try:
            s = raw.decode("utf-8").strip().splitlines()
            mk = s[0].strip(); mk = mk if mk else "model.pt"
            met = s[1].strip() if len(s) >= 2 else "metrics_tmp.json"
            ts = int(_head_mtime(_abs_key(mk)) or 0)
            return mk, met, ts
        except Exception:
            pass
    return None

# ------------- artefacts 加载 -------------
def load_model_by_key(key: str):
    """
    从 MinIO 读取并反序列化 torch 模型。
    key 可为绝对键（包含 /）或相对键（自动拼 MODEL_DIR/）。
    """
    k = _abs_key(key)
    raw = _read_bytes(k)
    if raw is None:
        raise FileNotFoundError(f"s3://{BUCKET}/{k} not found")
    mdl = torch.load(io.BytesIO(raw), map_location="cpu").eval()
    return mdl, raw

def load_scaler():
    raw = _read_bytes(f"{MODEL_DIR}/scaler.pkl")
    if raw is None:
        raise FileNotFoundError(f"s3://{BUCKET}/{MODEL_DIR}/scaler.pkl not found")
    bio = io.BytesIO(raw)
    return joblib.load(bio)

def load_selected_feats() -> List[str]:
    js = _read_json(f"{MODEL_DIR}/selected_feats.json")
    if not js or not isinstance(js, list):
        raise FileNotFoundError(f"s3://{BUCKET}/{MODEL_DIR}/selected_feats.json not found or invalid")
    return [str(c) for c in js]
