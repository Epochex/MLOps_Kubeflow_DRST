# shared/minio_helper.py
# ------------------------------------------------------------
# MinIO 上传/下载的简单包装
# ------------------------------------------------------------
import boto3, os, io, pandas as pd
from typing import Any
from .config import ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET

_sess = boto3.session.Session()
s3 = _sess.client(
    "s3",
    endpoint_url=f"http://{ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# ---------- helpers ----------

def load_csv(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return (
        pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)
          .replace({'<not counted>': None})
          .dropna()
    )

def save_bytes(key: str, data: bytes, content_type="application/octet-stream") -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

def save_np(key: str, arr: Any) -> None:
    import numpy as np
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(key, buf.read(), "application/npy")

def load_np(key: str):
    import numpy as np
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))
