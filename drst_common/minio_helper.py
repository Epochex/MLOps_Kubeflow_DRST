# drst_common/minio_helper.py
# MinIO client + PVC direct write + async upload; supports both Ingress/Cluster modes
import os
import io
import boto3
import threading
import concurrent.futures
from typing import Any

import pandas as pd
import numpy as np
from botocore.config import Config as BotoConfig

from .config import (
    ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET, DATA_DIR, RESULT_DIR,
    MINIO_SCHEME, MINIO_VERIFY_SSL, MINIO_CONSOLE_URL
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

def _async_upload(key: str, data: bytes, ctype: str):
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=ctype)
    except Exception as e:
        # Print console URL for easier troubleshooting
        print(f"[minio] upload failed: {key} | {e} | console={MINIO_CONSOLE_URL}")

def save_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """
    ① Write to PVC (/mnt/pvc/...) first so local processes can read immediately  
    ② Then upload asynchronously to MinIO to reduce blocking  
    Convention: store locally only as basename(key), to avoid permission/cleanup issues from nested paths
    """
    pvc_root = os.getenv("PVC_ROOT", "/mnt/pvc")
    local_base = os.path.basename(key)
    local_path = os.path.join(pvc_root, RESULT_DIR, local_base)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as fp:
        fp.write(data)

    _upload_pool.submit(_async_upload, key, data, content_type)

# ---------- Load / Save utilities ----------
def load_csv(key: str) -> pd.DataFrame:
    """Fetch CSV from MinIO into a local cache under DATA_DIR with the same basename, then read as DataFrame."""
    local = os.path.join(DATA_DIR, os.path.basename(key))
    if not os.path.exists(local):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(local, "wb") as f:
            f.write(obj["Body"].read())
    return (pd.read_csv(local, index_col=0)
              .replace({'<not counted>': None}).dropna())

def load_np(key: str) -> np.ndarray:
    """Fetch .npy from MinIO into a local cache under RESULT_DIR, then load with np.load."""
    local = os.path.join(RESULT_DIR, os.path.basename(key))
    if not os.path.exists(local):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(local, "wb") as f:
            f.write(obj["Body"].read())
    return np.load(local)

def save_np(key: str, arr: Any) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    save_bytes(key, buf.read(), "application/npy")
