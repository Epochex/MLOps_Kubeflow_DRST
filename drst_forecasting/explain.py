# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/explain.py
from __future__ import annotations
import os, io, json, time
import numpy as np
import shap
import matplotlib.pyplot as plt

import torch

from drst_common.config import MODEL_DIR, RESULT_DIR
from drst_common.minio_helper import save_bytes
from drst_common.resource_probe import start as start_probe

from .dataset import build_sliding_window
from .models import LSTMForecaster

LOOKBACK  = int(os.getenv("FORECAST_LOOKBACK", "48"))
HORIZON   = int(os.getenv("FORECAST_HORIZON",  "12"))
SAMPLE_N  = int(os.getenv("FORECAST_SHAP_N",   "256"))  # 解释采样数
HIDDEN    = int(os.getenv("FORECAST_HIDDEN",   "64"))
LAYERS    = int(os.getenv("FORECAST_LAYERS",   "1"))

device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model(in_dim: int) -> torch.nn.Module:
    raw = None
    try:
        from drst_common.minio_helper import s3, BUCKET
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/forecasting/model.pt")["Body"].read()
    except Exception as e:
        raise RuntimeError(f"cannot load forecasting model: {e}")
    model = torch.load(io.BytesIO(raw), map_location=device)
    model.eval()
    return model

def main():
    stop = start_probe("forecast_explain")
    print("[forecast.explain] start", flush=True)
    # 只取一个较小的样本子集做 KernelExplainer
    X, Y, feats = build_sliding_window(LOOKBACK, HORIZON, take_last_n=2000)
    in_dim = X.shape[-1]
    model = _load_model(in_dim)

    # 把时间维展平到模型期望的输入（这里直接喂 [T,D] → 模型 forward 再包一层）
    def f_predict(x2d: np.ndarray) -> np.ndarray:
        # x2d: [N, T*D]
        x = x2d.reshape(x2d.shape[0], LOOKBACK, in_dim).astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x).float().to(device)).cpu().numpy()  # [N, H]
        # 解释时通常取第一步（或平均），这里取第一步，形状为 [N]
        return y[:, 0:1]

    # 采样
    N = min(SAMPLE_N, len(X))
    sel = np.linspace(0, len(X)-1, num=N, dtype=int)
    Xs = X[sel].reshape(N, -1)  # [N, T*D]

    explainer = shap.KernelExplainer(f_predict, Xs[: min(64, N)])  # 小样本当背景
    sv = explainer.shap_values(Xs, nsamples="auto")  # [N, T*D] 的解释，返回 [N, 1] 的值向量
    sv = np.asarray(sv).reshape(N, -1)  # [N, T*D]

    # 保存数值
    bio = io.BytesIO(); np.save(bio, sv); bio.seek(0)
    save_bytes(f"{RESULT_DIR}/forecasting/shap_values.npy", bio.getvalue(), "application/npy")

    # 画 summary（把 [T*D] 展回 [T,D]，对 D 维做合并求和看重要性）
    sv_td = sv.reshape(N, LOOKBACK, in_dim)     # [N,T,D]
    imp = np.mean(np.abs(sv_td), axis=(0,1))    # [D]
    order = np.argsort(imp)[::-1]
    topk = min(15, len(feats))
    labels = [feats[i] for i in order[:topk]]
    vals   = imp[order[:topk]]

    plt.figure(figsize=(8, 4))
    plt.barh(range(topk), vals[::-1])
    plt.yticks(range(topk), labels[::-1])
    plt.title("Forecasting feature importance (SHAP, step-1)")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150); plt.close(); buf.seek(0)
    save_bytes(f"{RESULT_DIR}/forecasting/shap_summary.png", buf.read(), "image/png")

    md = ["# Forecasting SHAP Summary", "", f"- samples: {N}", f"- lookback: {LOOKBACK}", f"- horizon: {HORIZON}", "", "![shap](attachment://shap_summary.png)"]
    save_bytes(f"{RESULT_DIR}/forecasting/shap_summary.md", "\n".join(md).encode("utf-8"), "text/markdown")

    print("[forecast.explain] done", flush=True)
    stop()

if __name__ == "__main__":
    main()
