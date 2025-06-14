# MinIO 上传/下载 + 本地（PVC）缓存工具
import boto3, os, io, threading, concurrent.futures
import pandas as pd, numpy as np
from typing import Any
from .config import ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET, DATA_DIR, RESULT_DIR

# ---------- S3 客户端 ----------
_sess = boto3.session.Session()
s3 = _sess.client(
    "s3",
    endpoint_url=f"http://{ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

_upload_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _async_upload(key: str, data: bytes, ctype: str):
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=ctype)
    except Exception as e:
        print(f"[minio] upload failed: {key} | {e}")

def save_bytes(key: str, data: bytes,
               content_type: str = "application/octet-stream") -> None:
    """
    ① 先写 PVC（/mnt/pvc/...），供本地流程立刻读到  
    ② 再异步上传到 MinIO，减少阻塞
    """
    pvc_root = os.getenv("PVC_ROOT", "/mnt/pvc")
    local_base = os.path.basename(key)
    local_path = os.path.join(pvc_root, RESULT_DIR, local_base)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as fp:
        fp.write(data)

    # 后台线程池上传
    _upload_pool.submit(_async_upload, key, data, content_type)

# ---------- 加载 / 保存工具 ----------
def load_csv(key: str) -> pd.DataFrame:
    local = os.path.join(DATA_DIR, os.path.basename(key))
    if not os.path.exists(local):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(local, "wb") as f:
            f.write(obj["Body"].read())
    return (pd.read_csv(local, index_col=0)
              .replace({'<not counted>': None}).dropna())

def load_np(key: str) -> np.ndarray:
    local = os.path.join(RESULT_DIR, os.path.basename(key))
    if not os.path.exists(local):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(local, "wb") as f:
            f.write(obj["Body"].read())
    return np.load(local)

def save_np(key: str, arr: Any) -> None:
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(key, buf.read(), "application/npy")
