#!/usr/bin/env python3
# drst_inference/offline/train_offline.py
"""
End-to-end offline training:
- Read training set → clean/select features/standardize (features.prepare_dataset)
- Train baseline (linear) and adaptive (MLP)
- Evaluate MAE/RMSE/acc@0.15, write to models/{baseline_model.pt, model_*.pt, metrics_*.json, latest.txt}
- Log metrics into CSV/JSONL, and write /results/offline_done.flag
"""
from __future__ import annotations
import io
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .features import prepare_dataset
from .model import MLPRegressor, count_params

from drst_common.config import MODEL_DIR, RESULT_DIR
from drst_common.utils import calculate_accuracy_within_threshold
from drst_common.minio_helper import save_bytes
from drst_common.artefacts import write_latest
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata

# Hyperparameters (overridable via env)
TOPK          = int(os.getenv("OFFLINE_TOPK", "10")) or None  # 默认 Top-10 相关特征
VAL_FRAC      = float(os.getenv("OFFLINE_VAL_FRAC", "0.2"))
SEED          = int(os.getenv("OFFLINE_SEED", "42"))

EPOCHS_BASE   = int(os.getenv("OFFLINE_BASELINE_EPOCHS", "5"))
EPOCHS_ADAPT  = int(os.getenv("OFFLINE_ADAPT_EPOCHS",    "30"))
BATCH_SZ      = int(os.getenv("OFFLINE_BATCH",           "128"))
LR_BASE       = float(os.getenv("OFFLINE_BASELINE_LR",   "5e-3"))
LR_ADAPT      = float(os.getenv("OFFLINE_ADAPT_LR",      "1e-3"))
HIDDEN        = tuple(int(x) for x in os.getenv("OFFLINE_HIDDEN", "128,64").split(",")) if os.getenv("OFFLINE_HIDDEN") else (128, 64)
DROPOUT       = float(os.getenv("OFFLINE_DROPOUT", "0.0"))

device = "cuda" if torch.cuda.is_available() else "cpu"

def _train_regressor(model: nn.Module, Xtr: np.ndarray, Ytr: np.ndarray,
                     Xva: np.ndarray, Yva: np.ndarray,
                     epochs: int, batch: int, lr: float) -> Tuple[nn.Module, float]:
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float().view(-1,1))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    model = model.to(device)
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu())
        print(f"[offline] epoch {ep+1}/{epochs} loss={ep_loss/len(dl):.6f}")
    train_time = time.perf_counter() - t0

    # Validation metrics
    with torch.no_grad():
        pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
    mae = float(np.mean(np.abs(pv - Yva)))
    rmse= float(np.sqrt(np.mean((pv - Yva)**2)))
    acc15 = calculate_accuracy_within_threshold(pv, Yva, thr=0.15)
    return model, train_time

def main():
    touch_ready("offline", "trainer")

    # 1) Data and feature preparation (auto saves selected_feats.json / scaler.pkl)
    Xtr, Xva, Ytr, Yva, feats, _ = prepare_dataset(topk=TOPK, val_frac=VAL_FRAC, seed=SEED)
    in_dim = Xtr.shape[1]

    # 2) Baseline: single linear layer
    base = MLPRegressor(in_dim, hidden=(), dropout=0.0)
    base, base_time = _train_regressor(base, Xtr, Ytr, Xva, Yva, epochs=EPOCHS_BASE, batch=BATCH_SZ, lr=LR_BASE)

    # 3) Adaptive: multi-layer MLP
    adapt = MLPRegressor(in_dim, hidden=HIDDEN, dropout=DROPOUT)
    adapt, adapt_time = _train_regressor(adapt, Xtr, Ytr, Xva, Yva, epochs=EPOCHS_ADAPT, batch=BATCH_SZ, lr=LR_ADAPT)

    # 4) Evaluation (recompute validation metrics for logging)
    with torch.no_grad():
        pb = base(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
        pa = adapt(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()

    mae_b = float(np.mean(np.abs(pb - Yva)))
    rmse_b= float(np.sqrt(np.mean((pb - Yva)**2)))
    acc_b = calculate_accuracy_within_threshold(pb, Yva, thr=0.15)

    mae_a = float(np.mean(np.abs(pa - Yva)))
    rmse_a= float(np.sqrt(np.mean((pa - Yva)**2)))
    acc_a = calculate_accuracy_within_threshold(pa, Yva, thr=0.15)

    # 5) Save baseline_model.pt / model_*.pt + metrics_*.json + latest.txt
    # baseline
    buf_b = io.BytesIO(); torch.save(base.to("cpu"), buf_b)
    save_bytes(f"{MODEL_DIR}/baseline_model.pt", buf_b.getvalue(), "application/octet-stream")

    # adaptive (timestamped)
    ts = int(time.time())
    buf_a = io.BytesIO(); torch.save(adapt.to("cpu"), buf_a)
    model_key   = f"model_{ts}.pt"
    metrics_key = f"metrics_{ts}.json"
    metrics = {
        "acc@0.15": acc_a,
        "baseline_acc@0.15": acc_b,
        "mae": mae_a, "rmse": rmse_a,
        "baseline_mae": mae_b, "baseline_rmse": rmse_b,
        "train_rows": int(Xtr.shape[0]), "val_rows": int(Xva.shape[0]),
        "train_time_s": round(base_time + adapt_time, 4),
        "model_size_mb": round(len(buf_a.getvalue())/(1024*1024), 4),
        "hidden": list(HIDDEN), "dropout": float(DROPOUT),
    }
    write_latest(buf_a.getvalue(), metrics, model_key=model_key, metrics_key=metrics_key)

    # 6) Metrics logging
    log_metric(component="offline", event="train_summary",
               train_rows=int(Xtr.shape[0]), train_time_s=round(base_time+adapt_time, 4),
               mae=mae_a, rmse=rmse_a, accuracy=acc_a, model_size_mb=metrics["model_size_mb"])
    log_metric(component="offline", event="baseline_eval",
               mae=mae_b, rmse=rmse_b, accuracy=acc_b)

    # Final flag
    save_bytes(f"{RESULT_DIR}/offline_done.flag", f"ts={ts}\n".encode(), "text/plain")
    sync_all_metrics_to_minio()
    write_kfp_metadata()
    print(f"[offline] done. acc@0.15 baseline={acc_b:.4f} → adaptive={acc_a:.4f}, Δ={acc_a-acc_b:+.4f}")

if __name__ == "__main__":
    main()
