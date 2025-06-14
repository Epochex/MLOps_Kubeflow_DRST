import io
import json
import torch
import torch.nn as nn
from ml.model import DynamicMLP
from .minio_helper import s3, save_bytes
from .config import MODEL_DIR, BUCKET
from typing import Dict, Tuple
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    """
    将一段 raw bytes（可能是 torch.save(model) 或 state_dict）反序列化为 nn.Module。
    0) 如果直接存的是完整模型，立刻返回；
    1) 否则当成 state_dict，依次尝试用 last_config → 自动推断 → baseline_config 重建。
    """
    # 0) load 之后如果本身就是 Module，就直接用
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    # 1) 否则把它当成 state_dict 来处理
    state_dict = obj

    # helper: 给定 cfg，构建 DynamicMLP 并尝试 load_state_dict
    def try_with_cfg(cfg: dict) -> nn.Module | None:
        if not cfg or "hidden_layers" not in cfg:
            return None
        mdl = DynamicMLP(
            in_dim=input_dim,
            hidden_layers=tuple(cfg["hidden_layers"]),
            activation=cfg.get("activation", "relu")
        )
        try:
            mdl.load_state_dict(state_dict)
            return mdl.to(device).eval()
        except RuntimeError:
            return None

    # 推断输入维度：找到 net.0.weight 的第二维
    input_dim = next(
        v.shape[1]
        for k, v in state_dict.items()
        if k.endswith(".weight") and "net.0" in k
    )

    # 2) 优先用 dynamic retrain 的最新配置
    try:
        cfg_raw = _fetch("last_model_config.json")
        cfg = json.loads(cfg_raw.decode())
        mdl = try_with_cfg(cfg)
        if mdl is not None:
            return mdl
    except Exception:
        pass

    # 3) 保底：根据所有 weight 层的 out_features 推 hidden_layers
    weight_layers = [w for k, w in state_dict.items() if k.endswith(".weight")]
    hidden_layers_dyn = tuple(int(w.shape[0]) for w in weight_layers[:-1]) or (64,)
    mdl_auto = DynamicMLP(input_dim, hidden_layers_dyn, "relu")
    try:
        mdl_auto.load_state_dict(state_dict)
        return mdl_auto.to(device).eval()
    except Exception:
        pass

    # 4) 再尝试 baseline 的配置
    try:
        cfg_raw = _fetch("baseline_model_config.json")
        cfg = json.loads(cfg_raw.decode())
        mdl = try_with_cfg(cfg)
        if mdl is not None:
            return mdl
    except Exception:
        pass

    # 5) 最终兜底：直接返回 auto 重建的那个（一定能 load）
    print("[utils] fallback to weight-based rebuild (final)")
    mdl_auto.load_state_dict(state_dict)
    return mdl_auto.to(device).eval()



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
