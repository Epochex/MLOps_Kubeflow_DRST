import boto3, os, pathlib, io, pandas as pd

ENDPOINT   = os.getenv("MINIO_ENDPOINT",  "45.149.207.13:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY","minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY","minio123")
BUCKET     = os.getenv("MINIO_BUCKET",    "onvm-demo1")

_session = boto3.session.Session()
s3 = _session.client('s3',
        endpoint_url=f"http://{ENDPOINT}",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY)

def load_csv(key:str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return (pd.read_csv(io.BytesIO(obj['Body'].read()), index_col=0)
             .replace({'<not counted>': None, ' ': None})
             .dropna())

def save_bytes(key:str, data:bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=BUCKET, Key=key, Body=data,
                  ContentType=content_type)

def save_np(key:str, arr):
    import io, numpy as np
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(key, buf.read(), "application/npy")
