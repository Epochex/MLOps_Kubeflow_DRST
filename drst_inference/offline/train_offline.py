#!/usr/bin/env python3
from __future__ import annotations

import os, io, json, shutil, time
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

from drst_common.config import MODEL_DIR, RESULT_DIR, TARGET_COL, BUCKET, TRAIN_TRIGGER as CFG_TRAIN_TRIGGER
from drst_common.minio_helper import load_csv, save_bytes, s3
from drst_common.metric_logger import log_metric
from drst_common.utils import calculate_accuracy_within_threshold
from drst_common.resource_probe import start as start_probe

from drst_inference.offline.model import MLPRegressor
from drst_inference.offline.features import load_and_prepare

# =========================
# 参数（可被 env 覆盖）
# =========================
OFFLINE_KEY  = os.getenv("TRAIN_MINIO_KEY", os.getenv("OFFLINE_TRAIN_KEY", "datasets/combined.csv"))
RAND_KEY     = "datasets/random_rates.csv"
BRIDGE_N     = int(os.getenv("BRIDGE_N", "500"))
RAND_N       = int(os.getenv("RAND_N", "500"))
TOPK         = int(os.getenv("OFFLINE_TOPK", "10"))

# 重要：默认从 config 取值，环境变量仅覆盖
TRAIN_TRIGGER = int(os.getenv("TRAIN_TRIGGER", str(CFG_TRAIN_TRIGGER)))

LR           = float(os.getenv("OFFLINE_LR", "1e-2"))
BATCH_SIZE   = int(os.getenv("OFFLINE_BS", "16"))
MAX_EPOCH    = int(os.getenv("OFFLINE_MAX_EPOCH", "100"))
ES_PATIENCE  = int(os.getenv("OFFLINE_ES_PATIENCE", "10"))
SCHED_PAT    = int(os.getenv("OFFLINE_SCHED_PATIENCE", "5"))
MIN_LR       = float(os.getenv("OFFLINE_MIN_LR", "1e-5"))
VAL_FRAC     = float(os.getenv("OFFLINE_VAL_FRAC", "0.3"))
SEED_SPLIT   = int(os.getenv("OFFLINE_SPLIT_SEED", "0"))

ACC_THR_MAIN = float(os.getenv("ACC_THR_MAIN", "0.25"))  # 用于 combined 尾部桥接评估
ACC_THR_RAND = float(os.getenv("ACC_THR_RAND", "0.15"))  # 用于 random_rates 头部评估

device = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = "/tmp/offline_models"
os.makedirs(TMP_DIR, exist_ok=True)

def _save_as_baseline_and_model(model: nn.Module) -> None:
    local_base  = f"{TMP_DIR}/baseline_model.pt"
    local_model = f"{TMP_DIR}/model.pt"
    torch.save(model.eval().cpu(), local_base)
    shutil.copy(local_base, local_model)
    for local, key in [(local_base,  f"{MODEL_DIR}/baseline_model.pt"),
                       (local_model, f"{MODEL_DIR}/model.pt")]:
        with open(local, "rb") as f:
            save_bytes(key, f.read(), "application/octet-stream")
    # latest 指针（兼容 online 热加载）
    save_bytes(f"{MODEL_DIR}/latest.txt", b"model.pt\nmetrics_tmp.json", "text/plain")
    meta = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_bytes(f"{MODEL_DIR}/metrics_tmp.json", json.dumps(meta).encode(), "application/json")

def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
    return y

def _bridge_eval_and_dump(model: nn.Module, X_all: np.ndarray, y_all: np.ndarray, thr: float) -> float:
    n = min(BRIDGE_N, len(X_all))
    X_tail = X_all[-n:]; y_tail = y_all[-n:]
    y_pred = _predict(model, X_tail)
    acc = calculate_accuracy_within_threshold(y_tail, y_pred, thr)
    # 桥接产物（给后续画图/联动）
    for name, arr in [("bridge_true.npy", y_tail), ("bridge_pred.npy", y_pred)]:
        bio = io.BytesIO(); np.save(bio, arr); bio.seek(0)
        save_bytes(f"{RESULT_DIR}/{name}", bio.read(), "application/npy")
    return float(acc)

def _rand_head_eval(model: nn.Module, scaler, selected: List[str]) -> float:
    try:
        df = load_csv(RAND_KEY)
    except Exception:
        print(f"[offline] WARN: cannot load {RAND_KEY}, skip rand-head eval.")
        return 0.0
    # 预处理与对齐
    df = df.replace({"<not counted>": np.nan, r"^\s*$": np.nan}, regex=True)
    df = df.dropna(how="any").reset_index(drop=True)
    df = df.drop(columns=[c for c in ["Unnamed: 0","input_rate","latency"] if c in df.columns], errors="ignore")
    for c in selected:
        if c not in df.columns:
            df[c] = 0.0
    if TARGET_COL not in df.columns:
        print(f"[offline] WARN: {RAND_KEY} has no target column {TARGET_COL}; skip.")
        return 0.0
    df = df[selected + [TARGET_COL]]
    for c in selected:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    df = df.head(min(RAND_N, len(df)))
    if df.empty:
        return 0.0

    X = scaler.transform(df[selected].astype(np.float32).values)
    y = df[TARGET_COL].astype(np.float32).values
    y_pred = _predict(model, X)
    acc = calculate_accuracy_within_threshold(y, y_pred, ACC_THR_RAND)
    return float(acc)

def main():
    stop_probe = start_probe("offline")
    t0 = time.time()
    print(f"[offline] start — dataset={OFFLINE_KEY} TRAIN_TRIGGER={TRAIN_TRIGGER}", flush=True)

    # 1) 读全集 + 特征选择（Top-10）+ 标准化（对全集 fit）
    df_all, selected, scaler = load_and_prepare(OFFLINE_KEY, k=TOPK)
    X_all = scaler.transform(df_all[selected].astype(np.float32).values)
    y_all = df_all[TARGET_COL].astype(np.float32).values

    # 2) 划分 7:3（固定种子）
    Xtr, Xva, Ytr, Yva = train_test_split(X_all, y_all, test_size=VAL_FRAC, random_state=SEED_SPLIT)

    # 3) 模型与训练流程
    model = MLPRegressor(in_dim=Xtr.shape[1], hidden=(64, 32), act="relu").to(device)
    opt   = Adam(model.parameters(), lr=LR)
    lossf = nn.SmoothL1Loss()
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PAT, min_lr=MIN_LR)
    dl    = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                       batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_val = float("inf"); best_state: bytes | None = None; no_imp = 0

    # 4) 训练开关（含 baseline 缺失的自动兜底）
    do_train = (TRAIN_TRIGGER != 0)
    baseline_bytes: bytes | None = None
    if not do_train:
        try:
            baseline_bytes = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
        except Exception:
            print("[offline] baseline not found → switch to TRAIN mode for first run.", flush=True)
            do_train = True

    if do_train:
        print(f"[offline] training | lr={LR} bs={BATCH_SIZE} max_epoch={MAX_EPOCH} es_patience={ES_PATIENCE}", flush=True)
        for ep in range(1, MAX_EPOCH + 1):
            model.train()
            for xb, yb in dl:
                xb = xb.float().to(device); yb = yb.float().to(device).view(-1,1)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(xb), yb); loss.backward(); opt.step()
            # 验证
            with torch.no_grad():
                vpred = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
                val = float(np.mean((vpred - Yva) ** 2))
            sched.step(val)
            improved = val + 1e-9 < best_val
            if improved:
                best_val, no_imp = val, 0
                bio = io.BytesIO(); torch.save(model.to("cpu"), bio); best_state = bio.getvalue(); model.to(device)
            else:
                no_imp += 1
            if ep == 1 or ep % 5 == 0 or improved:
                print(f"[offline] epoch {ep:03d}/{MAX_EPOCH} | val={val:.6f} {'(*)' if improved else ''}", flush=True)
            if no_imp >= ES_PATIENCE:
                print(f"[offline] early-stop @ epoch {ep} (no_improve={no_imp})", flush=True); break

        if best_state is not None:
            model = torch.load(io.BytesIO(best_state), map_location=device)
        _save_as_baseline_and_model(model)
    else:
        # 不训练且 baseline 存在
        model = torch.load(io.BytesIO(baseline_bytes), map_location=device).eval().to(device)
        print("[offline] TRAIN_TRIGGER=0 → use existing baseline_model.pt", flush=True)

    # 5) 评估：combined 尾部（桥接） & random_rates 头部（acc@0.15）
    acc_bridge = _bridge_eval_and_dump(model, X_all, y_all, thr=ACC_THR_MAIN)
    print(f"[offline] [combined_tail{BRIDGE_N}] acc@{ACC_THR_MAIN:.2f} = {acc_bridge:.2f}%", flush=True)
    log_metric(component="offline", event="bridge_eval",
               n=min(BRIDGE_N, len(X_all)), **{f"acc@{ACC_THR_MAIN:.2f}".rstrip('0').rstrip('.'): round(acc_bridge, 2)})

    acc_rand = _rand_head_eval(model, scaler, selected)
    print(f"[offline] [random_rates_head{RAND_N}] acc@{ACC_THR_RAND:.2f} = {acc_rand:.2f}%", flush=True)
    log_metric(component="offline", event="random_rates_eval",
               n=RAND_N, **{f"acc@{ACC_THR_RAND:.2f}".rstrip('0').rstrip('.'): round(acc_rand, 2)})

    # 6) 结束与耗时
    wall_s = round(time.time() - t0, 3)
    log_metric(component="offline", event="train_done", wall_s=wall_s,
               mode=("TRAIN" if do_train else "SKIP"))
    print("[offline] done.", flush=True)

if __name__ == "__main__":
    main()
