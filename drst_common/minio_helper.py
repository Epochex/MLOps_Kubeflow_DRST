# drst_common/minio_helper.py
# MinIO client + PVC direct write + async upload; supports both Ingress/Cluster modes
from __future__ import annotations

import os
import io
import re
import boto3
import threading
import concurrent.futures
from typing import Any, Optional

import pandas as pd
import numpy as np
from botocore.config import Config as BotoConfig

from .config import (
    ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET,
    DATA_DIR, RESULT_DIR, MINIO_SCHEME, MINIO_VERIFY_SSL, MINIO_CONSOLE_URL
)

# ---------- S3 client ----------
_sess = boto3.session.Session()
_endpoint_url = f"{MINIO_SCHEME}://{ENDPOINT}"

# path-style for custom domains compatibility; keep retry config
_s3_config = BotoConfig(
    s3={"addressing_style": "path"},
    signature_version="s3v4",
    retries={"max_attempts": 10, "mode": "standard"},
)

# verify strategy:
# - http -> force False (avoid botocore complaints)
# - https -> prefer AWS_CA_BUNDLE path; otherwise fall back to MINIO_VERIFY_SSL bool
if MINIO_SCHEME == "http":
    _verify_arg: Any = False
else:
    _ca_bundle = os.environ.get("AWS_CA_BUNDLE")
    _verify_arg = _ca_bundle if (_ca_bundle and os.path.exists(_ca_bundle)) else MINIO_VERIFY_SSL

s3 = _sess.client(
    "s3",
    endpoint_url=_endpoint_url,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=_s3_config,
    verify=_verify_arg,
)

_upload_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _async_upload(key: str, data: bytes, ctype: str) -> None:
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=ctype)
    except Exception as e:
        # Print console URL for easier troubleshooting
        print(f"[minio] upload failed: {key} | {e} | console={MINIO_CONSOLE_URL}")

def save_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """
    ① 优先写入本地 PVC（/mnt/pvc/<RESULT_DIR>/<basename>），便于同 Pod 即时读取；
    ② 同步提交后台线程异步上传到 MinIO，减少阻塞。
    约定：本地仅以 basename(key) 命名，避免深层目录权限/清理问题。
    """
    pvc_root = os.getenv("PVC_ROOT", "/mnt/pvc")
    try:
        local_dir = os.path.join(pvc_root, RESULT_DIR)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(key))
        with open(local_path, "wb") as f:
            f.write(data)
    except Exception as e:
        # 本地写失败不阻断远端上传
        print(f"[minio] WARN local write failed: {e}")

    # 异步上传到 MinIO
    _upload_pool.submit(_async_upload, key, data, content_type)

def save_np(key: str, arr: np.ndarray, allow_pickle: bool = False) -> None:
    """保存 Numpy 数组到 MinIO（并本地留存一份）。"""
    bio = io.BytesIO()
    # np.save 不允许直接设置 content-type，这里由 save_bytes 处理
    np.save(bio, arr, allow_pickle=allow_pickle)
    save_bytes(key, bio.getvalue(), "application/octet-stream")

def _cache_local_path_for_csv(key: str) -> str:
    """
    统一的 CSV 本地缓存位置：<DATA_DIR>/<basename(key)>
    不创建子目录以简化权限和清理。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, os.path.basename(key))

def load_csv(key: str, *, encoding: Optional[str] = None) -> pd.DataFrame:
    """
    从 MinIO 拉取 CSV 到本地缓存再读。
    放宽清洗策略：不全表 dropna，只清理哨兵字符串和多余索引列。
    - 去掉类似 'Unnamed: 0' 的 pandas 索引残留列
    - 将字符串哨兵（如 '<not counted>'）置为 NaN，但不强制丢行
    """
    local = _cache_local_path_for_csv(key)

    # 若本地不存在则下载
    if not os.path.exists(local):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        with open(local, "wb") as f:
            f.write(obj["Body"].read())

    # 读取
    df = pd.read_csv(local, encoding=encoding)  # 不默认 index_col，避免误删列

    # 清理“多余索引列”：匹配 Unnamed: 0, Unnamed: 0.1 等
    unnamed_cols = [c for c in df.columns if re.match(r"^Unnamed:\s*\d+(\.\d+)?$", str(c))]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 替换常见哨兵字符串为 NaN（保留行，后续流程自行决定如何处理）
    if df.select_dtypes(include=["object"]).shape[1] > 0:
        df = df.replace({"<not counted>": np.nan})

    return df

def load_np(key: str, *, allow_pickle: bool = False) -> np.ndarray:
    """从 MinIO 读取 .npy / .npz 到内存并加载为 numpy 对象。"""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    bio = io.BytesIO(data)
    # 自动兼容 npy/npz：np.load 都能处理
    return np.load(bio, allow_pickle=allow_pickle)

# ——（可选）便捷函数：保存文本/JSON——
def save_text(key: str, text: str, encoding: str = "utf-8") -> None:
    save_bytes(key, text.encode(encoding), "text/plain; charset=utf-8")

def save_json(key: str, payload: Any) -> None:
    import json
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    save_bytes(key, data, "application/json")
