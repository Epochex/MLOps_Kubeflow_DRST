#!/usr/bin/env python3
from __future__ import annotations

import os
import io
import sys
import json
import shutil
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
if tuple(int(x) for x in np.__version__.split(".", 2)[:2]) >= (2, 0):
    print(
        f"[FATAL] NumPy {np.__version__} detected. 该镜像的 PyTorch 与 NumPy 2 不兼容；"
        "请在 requirements.txt 固定 numpy<2 并重建镜像。",
        file=sys.stderr,
    )
    sys.exit(2)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

from drst_common.minio_helper import load_csv, save_bytes, s3, BUCKET
from drst_common import config as _cfg
from drst_common.config import (
    MODEL_DIR, RESULT_DIR, TARGET_COL, EXCLUDE_COLS,
    OFFLINE_TRAIN_KEY, OFFLINE_EVAL_SOURCE_KEY, OFFLINE_EVAL_ROWS, OFFLINE_EVAL_TAKE, OFFLINE_EVAL_SEED
)
from drst_common.metric_logger import log_metric
from drst_common.utils import calculate_accuracy_within_threshold
from drst_inference.offline.model import MLPRegressor
from drst_inference.offline.features import prepare_dataset

TRAIN_TRIGGER   = int(os.getenv("TRAIN_TRIGGER", "1"))
OFFLINE_KEY     = os.getenv("TRAIN_MINIO_KEY", OFFLINE_TRAIN_KEY)

FAST_MAX_EPOCH  = int(os.getenv("FAST_MAX_EPOCH", "10"))
FAST_PATIENCE   = int(os.getenv("FAST_PATIENCE", "4"))
FAST_LR         = float(os.getenv("FAST_LR", "1e-3"))
FAST_BS         = int(os.getenv("FAST_BS", "64"))

FULL_MAX_EPOCH  = int(os.getenv("FULL_MAX_EPOCH", "100"))
FULL_PATIENCE   = int(os.getenv("FULL_PATIENCE", "10"))
FULL_LR         = float(os.getenv("FULL_LR", "1e-3"))
FULL_BS         = int(os.getenv("FULL_BS", "16"))

ACC_THR         = float(getattr(_cfg, "ACC_THR", float(os.getenv("ACC_THR", "0.25"))))

TMP_DIR         = "/tmp/offline_models"
os.makedirs(TMP_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_df_for_eval(df: pd.DataFrame, feat_list: List[str]) -> pd.DataFrame:
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")
    df = _coerce_numeric(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    keep = [c for c in feat_list if c in df.columns]
    for c in feat_list:
        if c not in df.columns:
            df[c] = 0.0
    df[feat_list] = (
        df[feat_list]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    cols = keep + [TARGET_COL] if TARGET_COL in df.columns else keep
    return df[cols]

def _evaluate(model: nn.Module,
              scaler: joblib,
              df: pd.DataFrame,
              feat_list: List[str],
              tag: str) -> float:
    X = scaler.transform(df[feat_list].values.astype(np.float32))
    y = df[TARGET_COL].astype(np.float32).values
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy().ravel()
    acc = calculate_accuracy_within_threshold(y, pred, ACC_THR)
    thr_str = f"{ACC_THR:.2f}".rstrip("0").rstrip(".")
    print(f"[offline] {tag} accuracy@{thr_str} = {acc:.2f}%", flush=True)
    return float(acc)

def _try_load_baseline() -> Optional[nn.Module]:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
        mdl = torch.load(io.BytesIO(raw), map_location=device).eval()
        print("[offline] found existing baseline_model.pt — using for pre-eval/skip.", flush=True)
        return mdl
    except Exception:
        return None

def _save_models_and_latest(model: nn.Module) -> None:
    local_base  = f"{TMP_DIR}/baseline_model.pt"
    local_model = f"{TMP_DIR}/model.pt"
    torch.save(model.eval().cpu(), local_base)
    shutil.copy(local_base, local_model)
    for local, key in [
        (local_base,  f"{MODEL_DIR}/baseline_model.pt"),
        (local_model, f"{MODEL_DIR}/model.pt"),
    ]:
        with open(local, "rb") as f:
            save_bytes(key, f.read(), "application/octet-stream")
    save_bytes(f"{MODEL_DIR}/latest.txt",
               b"model.pt\nmetrics_tmp.json",
               "text/plain")
    meta = {"timestamp_utc": datetime.utcnow().isoformat() + "Z"}
    save_bytes(f"{MODEL_DIR}/metrics_tmp.json",
               json.dumps(meta).encode(),
               "application/json")

def _write_bridge_artifacts(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for name, arr in [("bridge_true.npy", y_true), ("bridge_pred.npy", y_pred)]:
        buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
        save_bytes(f"{RESULT_DIR}/{name}", buf.read(), "application/npy")
    print("[offline] bridge artefacts uploaded ✔", flush=True)

def main():
    print(f"[offline] start — dataset={OFFLINE_KEY} TRAIN_TRIGGER={TRAIN_TRIGGER}", flush=True)

    Xtr, Xva, Ytr, Yva, SELECTED_FEATS, scaler = prepare_dataset(topk=10, val_frac=0.2, seed=42)

    eval_key = OFFLINE_EVAL_SOURCE_KEY or OFFLINE_KEY
    df_eval_all = load_csv(eval_key)
    df_eval_all = _clean_df_for_eval(df_eval_all, SELECTED_FEATS)
    n_eval = min(OFFLINE_EVAL_ROWS, len(df_eval_all))
    if OFFLINE_EVAL_TAKE == "tail":
        eval_df = df_eval_all.tail(n_eval).reset_index(drop=True)
    elif OFFLINE_EVAL_TAKE == "random":
        eval_df = df_eval_all.sample(n=n_eval, random_state=OFFLINE_EVAL_SEED).reset_index(drop=True)
    else:
        eval_df = df_eval_all.head(n_eval).reset_index(drop=True)
    print(f"[offline] eval set: key={eval_key}, take={OFFLINE_EVAL_TAKE}, rows={n_eval}", flush=True)

    pre_acc = None
    baseline_model = _try_load_baseline()
    if baseline_model is not None:
        pre_acc = _evaluate(baseline_model, scaler, eval_df, SELECTED_FEATS, "pre-eval")
        log_metric(component="offline", event="pre_eval", eval_acc=round(pre_acc or 0.0, 2), eval_rows=int(n_eval))

    do_train = TRAIN_TRIGGER != 0
    fast_mode = False
    if not do_train and baseline_model is None:
        do_train = True
        fast_mode = True
        print("[offline] no baseline available; switching to FAST training mode.", flush=True)

    if do_train:
        max_epoch = FAST_MAX_EPOCH if fast_mode else FULL_MAX_EPOCH
        patience  = FAST_PATIENCE  if fast_mode else FULL_PATIENCE
        lr        = FAST_LR        if fast_mode else FULL_LR
        bs        = FAST_BS        if fast_mode else FULL_BS

        print(f"[offline] training start | mode={'FAST' if fast_mode else 'FULL'} "
              f"| max_epoch={max_epoch} patience={patience} lr={lr} bs={bs}", flush=True)

        model = MLPRegressor(in_dim=Xtr.shape[1], hidden=(64, 32), act="relu").to(device)
        opt   = Adam(model.parameters(), lr=lr)
        sched = ReduceLROnPlateau(opt, factor=0.5, patience=3 if fast_mode else 5, min_lr=1e-5)
        lossf = nn.SmoothL1Loss()

        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                        batch_size=bs, shuffle=True, drop_last=False)

        best_val, best_state, no_imp = float("inf"), None, 0
        for ep in range(1, max_epoch + 1):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device).float()
                yb = yb.to(device).float().view(-1,1)
                opt.zero_grad(set_to_none=True)
                lossf(model(xb), yb).backward(); opt.step()

            with torch.no_grad():
                vpred = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
                val = float(np.mean((vpred - Yva) ** 2))
            sched.step(val)

            improved = val + 1e-9 < best_val
            if improved:
                best_val, best_state, no_imp = val, io.BytesIO(), 0
                torch.save(model.to("cpu"), best_state); best_state.seek(0); model.to(device)
            else:
                no_imp += 1

            if ep == 1 or ep % (3 if fast_mode else 5) == 0 or improved:
                print(f"[offline] epoch {ep:03d}/{max_epoch} | val={val:.6f} {'(*)' if improved else ''}", flush=True)
            if no_imp >= patience:
                print(f"[offline] early-stop @ epoch {ep} (no_improve={no_imp})", flush=True)
                break

        if best_state is not None:
            model = torch.load(best_state, map_location=device)
        print("[offline] training finished.", flush=True)

    else:
        model = baseline_model
        print("[offline] TRAIN_TRIGGER=0 and baseline exists → skip training.", flush=True)

    post_acc = _evaluate(model, scaler, eval_df, SELECTED_FEATS, "post-eval")
    log_metric(component="offline", event="post_eval", eval_acc=round(post_acc, 2), eval_rows=int(n_eval))

    _save_models_and_latest(model)

    Xe = scaler.transform(eval_df[SELECTED_FEATS].astype(np.float32).values)
    ye = eval_df[TARGET_COL].astype(np.float32).values
    with torch.no_grad():
        ype = model(torch.from_numpy(Xe).float().to(device)).cpu().numpy().ravel()
    _write_bridge_artifacts(y_true=ye, y_pred=ype)

    save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
               datetime.utcnow().isoformat().encode(), "text/plain")
    log_metric(component="offline", event="train_done",
               mode=("FAST" if do_train and fast_mode else ("FULL" if do_train else "SKIP")),
               max_epoch=(FAST_MAX_EPOCH if fast_mode else FULL_MAX_EPOCH) if do_train else 0)

    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
    print("[offline] done.", flush=True)

if __name__ == "__main__":
    main()
