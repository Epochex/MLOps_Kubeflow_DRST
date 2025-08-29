#!/usr/bin/env python3
# drst_drift/shap_analysis_placeholder.py
# Lightweight placeholder: when shap dependency is absent, use permutation importance as approximate feature importance;
# if shap is installed, automatically switch to KernelExplainer (may be slow, only sample a subset).
from __future__ import annotations
import os
import io
import json
import time
from typing import List, Dict, Tuple

import numpy as np

from drst_common.config import RESULT_DIR
from drst_common.minio_helper import load_np, save_bytes, s3
from drst_common.artefacts import load_selected_feats, load_scaler, load_model_by_key
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.config import BUCKET

# ---------- Tunable parameters ----------
MAX_SAMPLES = int(os.getenv("SHAP_MAX_SAMPLES", "512"))   # limit number of samples for speed
USE_KERNEL  = os.getenv("SHAP_USE_KERNEL", "auto").lower()  # auto/yes/no

pod_name = os.getenv("HOSTNAME", "shap")

def _align_to_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    if in_dim == X.shape[1]: return X
    if in_dim < X.shape[1]:  return X[:, :in_dim].copy()
    pad = np.zeros((X.shape[0], in_dim - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def _read_latest_columns() -> List[str] | None:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"{RESULT_DIR}/latest_batch.columns.json")["Body"].read()
        cols = json.loads(raw.decode())
        if isinstance(cols, list):
            return cols
    except Exception:
        pass
    return None

def _predict_fn(model, device: str):
    import torch
    model.eval().to(device)
    def f(x_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_np).float().to(device)
        with torch.no_grad():
            y = model(x).cpu().numpy().ravel()
        return y
    return f

def _perm_importance(model, X: np.ndarray, base_pred: np.ndarray) -> Dict[str, float]:
    """Approximate importance without labels: shuffle each column, measure mean absolute change in predictions."""
    imp: Dict[str, float] = {}
    n, d = X.shape
    rng = np.random.default_rng(0)
    Xc = X.copy()
    for j in range(d):
        orig = Xc[:, j].copy()
        rng.shuffle(Xc[:, j])
        pred = _predict_fn(model, device="cpu")(Xc)
        delta = np.mean(np.abs(pred - base_pred))
        imp[str(j)] = float(delta)
        Xc[:, j] = orig
    return imp

def _kernel_shap(model, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Only used if shap is available and USE_KERNEL=auto/yes; returns mean(|SHAP|) as importance."""
    try:
        import shap  # type: ignore
    except Exception:
        return {}
    if USE_KERNEL not in ("auto", "yes"):
        return {}

    print("[shap] using KernelExplainer (this may be slow)")
    f = _predict_fn(model, device="cpu")
    # sample subset (background + explain set)
    ns = min(MAX_SAMPLES, len(X))
    Xs = X[:ns]
    bg = Xs[: min(100, ns)]  # background
    expl = shap.KernelExplainer(f, bg)
    sv = expl.shap_values(Xs, nsamples="auto")
    if isinstance(sv, list):
        # For multi-output models, only take first output
        sv = sv[0]
    sv = np.asarray(sv)
    mean_abs = np.mean(np.abs(sv), axis=0).ravel()
    out: Dict[str, float] = {}
    for j, v in enumerate(mean_abs.tolist()):
        out[feature_names[j] if j < len(feature_names) else str(j)] = float(v)
    return out

def main():
    touch_ready("shap", pod_name)

    # 1) Data & features
    try:
        arr = load_np(f"{RESULT_DIR}/latest_batch.npy")
    except Exception as e:
        print(f"[shap] latest_batch.npy missing: {e}")
        return
    feat_names = load_selected_feats()
    scaler = load_scaler()
    cols = _read_latest_columns()

    d = len(feat_names)
    X = arr[:, :d].astype(np.float32) if arr.shape[1] >= d else arr.astype(np.float32)
    X = scaler.transform(X)
    # 2) Model & dimension alignment
    model, _ = load_model_by_key("model.pt")
    in_dim = getattr(model.net[0], "in_features", X.shape[1])
    X = _align_to_dim(X, in_dim)

    # Truncate samples
    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]

    # 3) Baseline prediction & permutation importance
    base_pred = _predict_fn(model, device="cpu")(X)
    imp_perm = _perm_importance(model, X, base_pred)

    # 4) If shap is available, supplement with KernelExplainer results
    imp_kernel = _kernel_shap(model, X, feat_names)

    # 5) Unified output: {feature: {perm_delta: x, kernel: y}}
    out: Dict[str, Dict[str, float]] = {}
    for j in range(len(X[0])):
        name = feat_names[j] if j < len(feat_names) else str(j)
        out[name] = {
            "perm_delta": float(imp_perm.get(str(j), 0.0)),
            "kernel": float(imp_kernel.get(name, 0.0)),
        }

    ts = int(time.time())
    save_bytes(f"{RESULT_DIR}/feature_importance_{ts}.json",
               json.dumps(out, ensure_ascii=False, indent=2).encode(),
               "application/json")
    save_bytes(f"{RESULT_DIR}/feature_importance_latest.json",
               json.dumps(out, ensure_ascii=False, indent=2).encode(),
               "application/json")

    log_metric(component="shap", event="ran", train_rows=len(X))
    sync_all_metrics_to_minio()
    write_kfp_metadata()
    print(f"[shap] importance written. n={len(X)}")

if __name__ == "__main__":
    main()
