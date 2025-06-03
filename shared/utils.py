# shared/utils.py
# ------------------------------------------------------------
# 通用工具函数：精度、JS、过滤、MinIO 存储
# ------------------------------------------------------------
import io, json, joblib, numpy as np
from scipy.spatial.distance import jensenshannon
from .minio_helper import save_bytes, s3, BUCKET
from .config import RESULT_DIR

# ---------- 评估 ----------
def calculate_accuracy_within_threshold(y_true, y_pred, thr=0.15):
    dist = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8)
    return (dist <= thr).mean() * 100

def js_divergence(p, q):
    return jensenshannon(p, q) ** 2

def filter_predictions(y_true, y_pred, threshold=0.1, accurate=True):
    err = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8)
    mask = err <= threshold if accurate else err > threshold
    return y_pred[mask], y_true[mask], np.where(mask)[0]

def filter_min(y_true, y_pred, min_value=500):
    mask = y_true >= min_value
    return y_pred[mask], y_true[mask], np.where(mask)[0]

# ---------- MinIO 存取 ----------
def save_pkl(path: str, obj):
    buf = io.BytesIO(); joblib.dump(obj, buf); buf.seek(0)
    save_bytes(path, buf.read())

def save_np(path: str, arr: np.ndarray):
    import numpy as np, io
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(path, buf.read(), "application/npy")

def load_np(key: str):
    import numpy as np, io
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))

def save_json(key: str, obj: dict):
    save_bytes(key, json.dumps(obj, ensure_ascii=False, indent=2).encode())

def load_json(key: str) -> dict:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode())
