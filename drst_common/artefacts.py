#!/usr/bin/env python3
# drst_common/artefacts.py
# 统一管理 models/ 下的 latest.txt / 模型 / metrics / scaler / 选特征 等工件
from __future__ import annotations
import io, json, hashlib
from typing import Tuple, Optional

import torch
import joblib

from .minio_helper import s3, save_bytes
from .config import BUCKET, MODEL_DIR
from .utils import _bytes_to_model

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def load_scaler():
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
    return joblib.load(io.BytesIO(raw))

def load_selected_feats() -> list[str]:
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
    return json.loads(raw.decode())

def load_model_by_key(model_key: str):
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{model_key}")["Body"].read()
    return _bytes_to_model(raw), raw

def read_latest() -> Optional[Tuple[str, str, dict]]:
    """返回 (model_key, metrics_key, metrics_dict)，若 latest 不存在返回 None。"""
    try:
        latest_raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/latest.txt")["Body"].read()
    except Exception:
        return None
    lines = latest_raw.decode().strip().splitlines()
    if len(lines) < 2:
        return None
    model_key, metrics_key = lines[0].strip(), lines[1].strip()
    metrics_raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{metrics_key}")["Body"].read()
    metrics = json.loads(metrics_raw.decode())
    return model_key, metrics_key, metrics

def write_latest(model_bytes: bytes, metrics: dict,
                 model_key: str = "model_tmp.pt",
                 metrics_key: str = "metrics_tmp.json") -> None:
    save_bytes(f"{MODEL_DIR}/{model_key}", model_bytes, "application/octet-stream")
    save_bytes(f"{MODEL_DIR}/{metrics_key}", json.dumps(metrics).encode(), "application/json")
    save_bytes(f"{MODEL_DIR}/latest.txt", f"{model_key}\n{metrics_key}".encode(), "text/plain")
