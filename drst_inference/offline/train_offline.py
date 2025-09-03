#!/usr/bin/env python3
# drst_inference/offline/train_offline.py
"""
离线训练脚本（已移除旧项目 shared/* 与 ml/* 依赖）
- 从 MinIO 读取 TRAIN_MINIO_KEY（默认 datasets/combined.csv）
- 通过 features.prepare_dataset() 选择特征并写入 models/selected_feats.json & models/scaler.pkl
- 训练 MLPRegressor（或在 TRAIN_TRIGGER=0 且存在 baseline 时跳过训练）
- 写出 models/baseline_model.pt、models/model.pt、models/latest.txt、models/metrics_tmp.json
- 产出 results/bridge_true.npy、bridge_pred.npy（供绘图）
"""

from __future__ import annotations
import os
import io
import json
import shutil
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

from drst_common.minio_helper import load_csv, save_bytes, s3, BUCKET
from drst_common.config       import MODEL_DIR, RESULT_DIR, TARGET_COL, EXCLUDE_COLS
from drst_common.metric_logger import log_metric
from drst_common.utils        import calculate_accuracy_within_threshold
from drst_inference.offline.model import MLPRegressor
from drst_inference.offline.features import prepare_dataset

# ========================= 运行参数（可用环境变量覆盖） =========================
TRAIN_TRIGGER   = int(os.getenv("TRAIN_TRIGGER", "1"))                     # 1 训练 / 0 尝试复用 baseline
OFFLINE_KEY     = os.getenv("TRAIN_MINIO_KEY", "datasets/combined.csv")    # 训练集在 MinIO 的键
TEST_KEY        = os.getenv("TEST_MINIO_KEY",  "datasets/random_rates.csv")# 评估集（随机）在 MinIO 的键

# “快速模式”（冷启动时使用，或你想要更快出模型）
FAST_MAX_EPOCH  = int(os.getenv("FAST_MAX_EPOCH", "10"))
FAST_PATIENCE   = int(os.getenv("FAST_PATIENCE", "4"))
FAST_LR         = float(os.getenv("FAST_LR", "1e-3"))
FAST_BS         = int(os.getenv("FAST_BS", "64"))

# “完整模式”（正常训练）
FULL_MAX_EPOCH  = int(os.getenv("FULL_MAX_EPOCH", "100"))
FULL_PATIENCE   = int(os.getenv("FULL_PATIENCE", "10"))
FULL_LR         = float(os.getenv("FULL_LR", "1e-2"))
FULL_BS         = int(os.getenv("FULL_BS", "16"))

# 评估子集配置
BRIDGE_N        = int(os.getenv("BRIDGE_N", "500"))
RAND_N          = int(os.getenv("RAND_N",   "500"))

TMP_DIR         = "/tmp/offline_models"
os.makedirs(TMP_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================= 小工具函数 =========================
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_df_for_eval(df: pd.DataFrame, feat_list: List[str]) -> pd.DataFrame:
    # 去除无关列，保证特征齐全
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")
    df = _coerce_numeric(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    keep = [c for c in feat_list if c in df.columns]
    # 如特征缺失，用 0 填（避免报错）
    for c in feat_list:
        if c not in df.columns:
            df[c] = 0.0
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
    acc = calculate_accuracy_within_threshold(y, pred, 0.15)
    print(f"[offline] {tag} accuracy@0.15 = {acc:.2f}%", flush=True)
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
    # baseline_model.pt & model.pt
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

    # latest.txt + metrics_tmp.json
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

# ========================= 主流程 =========================
def main():
    print(f"[offline] start — dataset={OFFLINE_KEY} TRAIN_TRIGGER={TRAIN_TRIGGER}", flush=True)

    # 1) 特征准备（写出 selected_feats.json & scaler.pkl）
    #    - prepare_dataset 会从 TRAIN_MINIO_KEY 读取全集，选择特征并写出工件
    #    - 我们拿到训练/验证数组和列表 & scaler
    Xtr, Xva, Ytr, Yva, SELECTED_FEATS, scaler = prepare_dataset(topk=10, val_frac=0.2, seed=42)

    # 2) 构造桥接评估集（全集的最后 BRIDGE_N 行）
    df_all = load_csv(OFFLINE_KEY)
    df_all = _clean_df_for_eval(df_all, SELECTED_FEATS)
    bridge_df = df_all.tail(BRIDGE_N).reset_index(drop=True)

    # 3) 读取随机评估集
    try:
        df_rand = load_csv(TEST_KEY)
        df_rand = _clean_df_for_eval(df_rand, SELECTED_FEATS)
        rand_sub = df_rand.head(min(RAND_N, len(df_rand))).reset_index(drop=True)
    except Exception as e:
        print(f"[offline] WARN random set not available: {e}", flush=True)
        rand_sub = bridge_df.iloc[:min(200, len(bridge_df))].copy()

    # 4) 若有 baseline，则先做预评估
    pre_acc_bridge = pre_acc_rand = None
    baseline_model = _try_load_baseline()
    if baseline_model is not None:
        pre_acc_bridge = _evaluate(baseline_model, scaler, bridge_df, SELECTED_FEATS, "pre-bridge")
        pre_acc_rand   = _evaluate(baseline_model, scaler, rand_sub , SELECTED_FEATS, "pre-random")
        log_metric(component="offline", event="pre_eval",
                   bridge_acc=round(pre_acc_bridge or 0.0, 2),
                   random_acc=round(pre_acc_rand or 0.0, 2))

    # 5) 是否训练
    do_train = TRAIN_TRIGGER != 0
    fast_mode = False
    if not do_train and baseline_model is None:
        do_train = True
        fast_mode = True
        print("[offline] no baseline available; switching to FAST training mode.", flush=True)

    # 6) 训练或复用 baseline
    if do_train:
        max_epoch = FAST_MAX_EPOCH if fast_mode else FULL_MAX_EPOCH
        patience  = FAST_PATIENCE  if fast_mode else FULL_PATIENCE
        lr        = FAST_LR        if fast_mode else FULL_LR
        bs        = FAST_BS        if fast_mode else FULL_BS

        print(f"[offline] training start | mode={'FAST' if fast_mode else 'FULL'} "
              f"| max_epoch={max_epoch} patience={patience} lr={lr} bs={bs}", flush=True)

        # 用 prepare_dataset 的 Xtr/Xva/Ytr/Yva
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
                val = float(np.mean((vpred - Yva) ** 2))  # 用 MSE 作为 early-stop 指标
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

    # 7) 训练后评估 & 指标记录
    post_acc_bridge = _evaluate(model, scaler, bridge_df, SELECTED_FEATS, "post-bridge")
    post_acc_rand   = _evaluate(model, scaler, rand_sub , SELECTED_FEATS, "post-random")
    log_metric(component="offline", event="post_eval",
               bridge_acc=round(post_acc_bridge, 2),
               random_acc=round(post_acc_rand, 2))

    # 8) 写出模型及 latest
    _save_models_and_latest(model)

    # 9) 写 bridge_true / bridge_pred （用“post”模型）
    Xb = scaler.transform(bridge_df[SELECTED_FEATS].astype(np.float32).values)
    yb = bridge_df[TARGET_COL].astype(np.float32).values
    with torch.no_grad():
        ypb = model(torch.from_numpy(Xb).float().to(device)).cpu().numpy().ravel()
    _write_bridge_artifacts(y_true=yb, y_pred=ypb)

    # 10) 元信息 & KFP 占位输出
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
