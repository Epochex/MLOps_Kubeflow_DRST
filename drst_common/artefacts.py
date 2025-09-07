#!/usr/bin/env python3
# drst_common/artefacts.py
# 统一管理 models/ 下的 latest.txt / 模型 / metrics / scaler / 选特征 等工件
from __future__ import annotations
import io
import json
import hashlib
from typing import Tuple, Optional

import torch
import joblib

from .minio_helper import s3, save_bytes
from .config import BUCKET, MODEL_DIR

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def load_scaler():
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
    return joblib.load(io.BytesIO(raw))

def load_selected_feats() -> list[str]:
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
    return json.loads(raw.decode())

def load_model_by_key(model_key: str):
    """
    从 MinIO 读取模型权重并反序列化为 torch.nn.Module。
    返回 (model, raw_bytes)
    """
    key = model_key if model_key.startswith(MODEL_DIR + "/") else f"{MODEL_DIR}/{model_key}"
    raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    model = torch.load(io.BytesIO(raw), map_location="cpu").eval()
    return model, raw

def read_latest() -> Optional[Tuple[str, str, Optional[str]]]:
    """
    读取 models/latest.txt → (model_key, metrics_key, ts?)
    latest.txt 按行存储：
      第1行：模型键，如 model.pt 或 model_1700000000.pt
      第2行：指标键，如 metrics_tmp.json 或 metrics_1700000000.json
      第3行：可选的时间戳/备注
    """
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/latest.txt")["Body"].read()
        lines = [x.strip() for x in raw.decode().splitlines() if x.strip()]
        if not lines:
            return None
        model_key   = lines[0]
        metrics_key = lines[1] if len(lines) >= 2 else ""
        ts          = lines[2] if len(lines) >= 3 else None
        # 允许省略前缀
        if not model_key.startswith(MODEL_DIR + "/"):
            model_key = f"{MODEL_DIR}/{model_key}"
        if metrics_key and not metrics_key.startswith(MODEL_DIR + "/"):
            metrics_key = f"{MODEL_DIR}/{metrics_key}"
        return model_key, metrics_key, ts
    except Exception:
        return None

def write_latest(model_bytes: bytes,
                 metrics: dict,
                 model_key: str = "model.pt",
                 metrics_key: str = "metrics.json",
                 ts: Optional[str] = None) -> None:
    """
    将模型与指标写入 MinIO 并刷新 latest.txt。
    - model_key / metrics_key 可传绝对键或文件名（自动补上 models/ 前缀）
    - 第三行可写 ts（不传则留空）
    """
    mkey = model_key if model_key.startswith(MODEL_DIR + "/") else f"{MODEL_DIR}/{model_key}"
    jkey = metrics_key if metrics_key.startswith(MODEL_DIR + "/") else f"{MODEL_DIR}/{metrics_key}"

    # 写模型
    save_bytes(mkey, model_bytes, "application/octet-stream")
    # 写指标
    save_bytes(jkey, json.dumps(metrics, ensure_ascii=False, indent=2).encode(), "application/json")

    # 刷新 latest.txt
    payload = f"{mkey.split('/',1)[1]}\n{jkey.split('/',1)[1]}"
    if ts:
        payload += f"\n{ts}"
    save_bytes(f"{MODEL_DIR}/latest.txt", payload.encode(), "text/plain")
