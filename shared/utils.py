# shared/utils.py
# ------------------------------------------------------------
# 通用工具函数：精度、JS、过滤、MinIO 存储
# ------------------------------------------------------------
import io, json, joblib, numpy as np
import torch                          # ★ 新增
import torch.nn as nn                 # ★ 新增
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon

from .minio_helper import save_bytes, s3, BUCKET
from .config       import RESULT_DIR, MODEL_DIR 
device = "cuda" if torch.cuda.is_available() else "cpu" 
# ---------- MinIO fetch（本地 notebook 也能用） --------------------------
def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET,
                         Key=f"{MODEL_DIR}/{key}")["Body"].read()

# ---------- 把权重 bytes 还原成 DynamicMLP -------------------------------
def _bytes_to_model(raw: bytes) -> nn.Module:
    from ml.model import DynamicMLP                             # 延迟导入防循环
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    state_dict = obj
    # 先看有无显式 config
    try:
        cfg = json.loads(_fetch("last_model_config.json").decode())
        hidden_layers = tuple(cfg["hidden_layers"])
        activation    = cfg.get("activation", "relu")
    except Exception:
        weight_layers = sorted(v for k, v in state_dict.items()
                               if k.startswith("net.") and k.endswith(".weight"))
        hidden_layers = tuple(int(w.shape[0]) for w in weight_layers[:-1]) or (64,)
        activation    = "relu"

    in_dim = next(v for k, v in state_dict.items()
                  if k.startswith("net.0") and k.endswith("weight")).shape[1]
    model = DynamicMLP(in_dim, hidden_layers, activation)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

# ---------- 评估 ---------------------------------------------------------
def calculate_accuracy_within_threshold(y_true, y_pred, thr: float = 0.15) -> float:
    """|err|/true <= thr 的比例 ×100%"""
    dist = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8)
    return (dist <= thr).mean() * 100

def calculate_accuracy_cdf(y_true, y_pred,
                           max_threshold: float = 1.0,
                           step: float = 0.1) -> Dict[float, float]:
    """返回各阈值上的准确率 CDF，用于画阶梯图"""
    thresholds = np.arange(0, max_threshold + 1e-9, step)
    return {
        thr: calculate_accuracy_within_threshold(y_true, y_pred, thr)
        for thr in thresholds
    }

def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence (squared)"""
    return jensenshannon(p, q) ** 2

def calculate_js_divergence_between_datasets(
        df_a, df_b, features, bins: int = 30) -> Tuple[Dict[str, float], float]:
    """
    计算两个 DataFrame 在指定特征上的 JS divergence。
    返回 (逐列 JS, 平均 JS)
    """
    res = {}
    for feat in features:
        if feat not in df_a.columns or feat not in df_b.columns:
            continue
        rng = (
            min(df_a[feat].min(), df_b[feat].min()),
            max(df_a[feat].max(), df_b[feat].max()),
        )
        p_hist, _ = np.histogram(df_a[feat], bins=bins, range=rng, density=True)
        q_hist, _ = np.histogram(df_b[feat], bins=bins, range=rng, density=True)
        res[feat] = js_divergence(p_hist, q_hist)
    avg = float(np.mean(list(res.values()))) if res else 0.0
    return res, avg

# ---------- 预测过滤 -----------------------------------------------------
def filter_predictions(y_true, y_pred,
                       threshold: float = 0.1,
                       accurate: bool = True):
    err  = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8)
    mask = err <= threshold if accurate else err > threshold
    return y_pred[mask], y_true[mask], np.where(mask)[0]

def filter_min(y_true, y_pred, min_value: float = 500):
    mask = y_true >= min_value
    return y_pred[mask], y_true[mask], np.where(mask)[0]

# ---------- MinIO 存取 ---------------------------------------------------
def save_pkl(path: str, obj):
    buf = io.BytesIO(); joblib.dump(obj, buf); buf.seek(0)
    save_bytes(path, buf.read())

def save_np(path: str, arr: np.ndarray):
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(path, buf.read(), "application/npy")

def load_np(key: str):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))

def save_json(key: str, obj: dict):
    save_bytes(key, json.dumps(obj, ensure_ascii=False, indent=2).encode())

def load_json(key: str) -> dict:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode())
