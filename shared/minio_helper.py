# shared/minio_helper.py
# ------------------------------------------------------------
# MinIO 上传/下载的简单包装（支持本地缓存）
# ------------------------------------------------------------
import boto3
import os
import io
import pandas as pd
import numpy as np
from typing import Any
from .config import ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET, DATA_DIR, RESULT_DIR

# 初始化 S3 客户端
_sess = boto3.session.Session()
s3 = _sess.client(
    "s3",
    endpoint_url=f"http://{ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

def load_csv(key: str) -> pd.DataFrame:
    """
    从本地缓存加载 CSV；如不存在则从 MinIO 拉取并缓存。
    key: MinIO 中的对象 Key（例如 'datasets/combined.csv'）。
    """
    local_base = os.path.basename(key)
    local_path = os.path.join(DATA_DIR, local_base)
    if os.path.exists(local_path):
        return (
            pd.read_csv(local_path, index_col=0)
              .replace({'<not counted>': None})
              .dropna()
        )
    # 从 MinIO 拉取并缓存到本地
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    data = obj["Body"].read()
    with open(local_path, "wb") as f:
        f.write(data)
    return (
        pd.read_csv(local_path, index_col=0)
          .replace({'<not counted>': None})
          .dropna()
    )

def load_np(key: str) -> np.ndarray:
    """
    从本地缓存加载 NumPy 文件；如不存在则从 MinIO 拉取并缓存。
    key: MinIO 中的对象 Key（例如 'results/bridge_true.npy'）。
    """
    local_base = os.path.basename(key)
    local_path = os.path.join(RESULT_DIR, local_base)
    if os.path.exists(local_path):
        return np.load(local_path)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    data = obj["Body"].read()
    with open(local_path, "wb") as f:
        f.write(data)
    return np.load(io.BytesIO(data))

def save_bytes(key: str, data: bytes, content_type="application/octet-stream") -> None:
    """
    上传字节数据到 MinIO，并写入本地缓存（供后续调试或其他组件读取）。
    """
    # 先上传到 MinIO
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    # 再写入本地
    local_base = os.path.basename(key)
    local_path = os.path.join(RESULT_DIR, local_base)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)

def save_np(key: str, arr: Any) -> None:
    """
    保存 NumPy 数组到 MinIO，并写本地缓存。
    """
    import numpy as _np, io as _io
    buf = _io.BytesIO()
    _np.save(buf, arr)
    buf.seek(0)
    save_bytes(key, buf.read(), "application/npy")
