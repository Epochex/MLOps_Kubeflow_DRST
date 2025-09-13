#!/usr/bin/env python3
# drst_common/minio_helper.py
from __future__ import annotations
import os
import io
import json
import time
from typing import Optional

import numpy as np
import pandas as pd
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

try:
    from . import config as _cfg
except Exception:
    _cfg = None  # 允许完全无 config.py 的极端情况

def _get(name: str, *alts, env: Optional[str] = None, default=None):
    # 先从 config.py 取（按顺序尝试），再取同名环境变量，最后 default
    for key in (name,) + tuple(alts):
        if _cfg is not None and hasattr(_cfg, key):
            return getattr(_cfg, key)
    if env:  # 显式 env 名称
        v = os.getenv(env)
        if v is not None:
            return v
    v = os.getenv(name)
    if v is not None:
        return v
    return default

MINIO_SCHEME   = _get("MINIO_SCHEME", default="http")
MINIO_ENDPOINT = _get("MINIO_ENDPOINT", "ENDPOINT", default="minio-service.kubeflow.svc.cluster.local:9000")
MINIO_BUCKET   = _get("MINIO_BUCKET", "BUCKET", default="mlpipeline")
MINIO_ACCESS   = _get("MINIO_ACCESS_KEY", "ACCESS_KEY", default=os.getenv("MINIO_ACCESS_KEY", "minio"))
MINIO_SECRET   = _get("MINIO_SECRET_KEY", "SECRET_KEY", default=os.getenv("MINIO_SECRET_KEY", "minio123"))
MINIO_REGION   = _get("MINIO_REGION", default="us-east-1")

BUCKET = str(MINIO_BUCKET)

_endpoint_url = os.getenv("MINIO_ENDPOINT_URL") or f"{MINIO_SCHEME}://{MINIO_ENDPOINT}".rstrip("/")

# ---- boto3 S3 客户端 ----
s3 = boto3.client(
    "s3",
    endpoint_url=_endpoint_url,
    aws_access_key_id=MINIO_ACCESS,
    aws_secret_access_key=MINIO_SECRET,
    region_name=MINIO_REGION,
    config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
)

# ---- 便捷 I/O（带轻量重试）----
def _retry(fn, *, tries: int = 3, delay: float = 0.5):
    last = None
    for _ in range(max(1, tries)):
        try:
            return fn()
        except (BotoCoreError, ClientError) as e:
            last = e
            time.sleep(delay)
        except Exception as e:
            last = e
            time.sleep(delay)
    if last:
        raise last

def save_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    def _put():
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    _retry(_put)

def save_np(key: str, arr: np.ndarray, allow_pickle: bool = False) -> None:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=allow_pickle)
    bio.seek(0)
    save_bytes(key, bio.read(), "application/npy")

def load_np(key: str, allow_pickle: bool = False) -> np.ndarray:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()), allow_pickle=allow_pickle)

def load_csv(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(obj["Body"])
    for c in list(df.columns):
        if str(c).startswith("Unnamed:"):
            df = df.drop(columns=[c])
    df = df.replace({"<not counted>": np.nan})
    return df

# ---- 小工具：调试信息（可选）----
def _debug_dump_config() -> dict:
    return {
        "endpoint_url": _endpoint_url,
        "bucket": BUCKET,
        "have_env_access": bool(MINIO_ACCESS),
        "have_env_secret": bool(MINIO_SECRET),
    }

if __name__ == "__main__":
    print(json.dumps(_debug_dump_config(), indent=2))
