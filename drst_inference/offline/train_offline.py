#!/usr/bin/env python3
from __future__ import annotations

import os, io, sys, json, shutil, time
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np
if tuple(int(x) for x in np.__version__.split(".", 2)[:2]) >= (2, 0):
    print(
        f"[FATAL] NumPy {np.__version__} detected. 该镜像的 PyTorch 与 NumPy 2 不兼容；"
        "请在 requirements.txt 固定 numpy<2 并重建镜像。",
        file=sys.stderr,
    ); sys.exit(2)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

from drst_common.minio_helper import load_csv, save_bytes, s3, BUCKET
from drst_common import config as _cfg
from drst_common.config import MODEL_DIR, RESULT_DIR, TARGET_COL, EXCLUDE_COLS, OFFLINE_TRAIN_KEY
from drst_common.metric_logger import log_metric
from drst_common.utils import calculate_accuracy_within_threshold
from drst_common.resource_probe import start as start_probe
from drst_inference.offline.model import MLPRegressor
from drst_inference.offline.features import prepare_dataset

TRAIN_TRIGGER   = int(os.getenv("TRAIN_TRIGGER", "1"))
OFFLINE_KEY     = os.getenv("TRAIN_MINIO_KEY", OFFLINE_TRAIN_KEY)

# 训练超参（保持原样，支持 FAST/FULL）
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

# —— 本次新增：两组离线评估的“固定方案” —— #
EVAL_PLAN: List[Dict] = [
    {"tag": "combined500",      "key": "datasets/combined.csv",                               "rows": 500,  "take": "random", "seed": 42},
    {"tag": "random_rates500",  "key": "datasets/random_rates.csv",                           "rows": 500,  "take": "random", "seed": 42},
]
# ------------------------------------------------------- #

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_df_for_eval(df: pd.DataFrame, feat_list: List[str]) -> pd.DataFrame:
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")
    df = _coerce_numeric(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    keep = [c for c in feat_list if c in df.columns]
    for c in feat_list:
        if c not in df.columns: df[c] = 0.0
    df[feat_list] = df[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = keep + [TARGET_COL] if TARGET_COL in df.columns else keep
    return df[cols]

def _evaluate(model: nn.Module, scaler: joblib, df: pd.DataFrame, feat_list: List[str], tag: str) -> float:
    X = scaler.transform(df[feat_list].values.astype(np.float32))
    y = df[TARGET_COL].astype(np.float32).values
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy().ravel()
    acc = calculate_accuracy_within_threshold(y, pred, ACC_THR)
    thr_str = f"{ACC_THR:.2f}".rstrip("0").rstrip(".")
    print(f"[offline] {tag} accuracy@{thr_str} = {acc:.2f}%", flush=True)
    return float(acc)

def _try_load_existing_baseline() -> Optional[nn.Module]:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
        mdl = torch.load(io.BytesIO(raw), map_location=device).eval()
        print("[offline] found existing baseline_model.pt — using for pre-eval.", flush=True)
        return mdl
    except Exception:
        return None

def _save_models_and_latest(model: nn.Module) -> None:
    local_base  = f"{TMP_DIR}/baseline_model.pt"
    local_model = f"{TMP_DIR}/model.pt"
    torch.save(model.eval().cpu(), local_base)
    shutil.copy(local_base, local_model)
    for local, key in [(local_base,  f"{MODEL_DIR}/baseline_model.pt"), (local_model, f"{MODEL_DIR}/model.pt")]:
        with open(local, "rb") as f:
            save_bytes(key, f.read(), "application/octet-stream")
    save_bytes(f"{MODEL_DIR}/latest.txt", b"model.pt\nmetrics_tmp.json", "text/plain")
    meta = {"timestamp_utc": datetime.utcnow().isoformat() + "Z"}
    save_bytes(f"{MODEL_DIR}/metrics_tmp.json", json.dumps(meta).encode(), "application/json")

def _write_bridge_artifacts(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for name, arr in [("bridge_true.npy", y_true), ("bridge_pred.npy", y_pred)]:
        bio = io.BytesIO(); np.save(bio, arr); bio.seek(0)
        save_bytes(f"{RESULT_DIR}/{name}", bio.read(), "application/npy")
    print("[offline] bridge artefacts uploaded ✔", flush=True)

def _subset(df: pd.DataFrame, rows: int, take: str, seed: Optional[int]) -> pd.DataFrame:
    n = min(int(rows), len(df))
    take = (take or "head").lower()
    if take == "tail":   return df.tail(n).reset_index(drop=True)
    if take == "random": return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)

def _multi_eval(
    model_new: nn.Module,
    model_base: Optional[nn.Module],
    scaler, feats: List[str]
) -> None:
    """执行 EVAL_PLAN；对每个数据集同时算 new/base 两个模型的 acc，并打点。"""
    thr_str = f"{ACC_THR:.2f}".rstrip("0").rstrip(".")
    for spec in EVAL_PLAN:
        tag, key, rows, take, seed = spec["tag"], spec["key"], spec["rows"], spec["take"], spec.get("seed")
        try:
            df_all = load_csv(key)
            df_all = _clean_df_for_eval(df_all, feats)
            df_eval = _subset(df_all, rows, take, seed)
        except Exception as e:
            print(f"[offline] eval load failed @ {tag} ({key}): {e}")
            continue

        # new
        X = scaler.transform(df_eval[feats].astype(np.float32).values)
        y = df_eval[TARGET_COL].astype(np.float32).values
        with torch.no_grad():
            pred_new = model_new(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
        acc_new = calculate_accuracy_within_threshold(y, pred_new, ACC_THR)

        # base（如存在）
        acc_base = None
        if model_base is not None:
            with torch.no_grad():
                pred_base = model_base(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
            acc_base = calculate_accuracy_within_threshold(y, pred_base, ACC_THR)

        print(f"[offline] [{tag}] new_acc@{thr_str}={acc_new:.2f}%"
              + (f" | base_acc@{thr_str}={acc_base:.2f}%" if acc_base is not None else ""), flush=True)

        # metrics
        log_metric(component="offline", event=f"offline_eval_{tag}",
                   eval_rows=int(len(df_eval)),
                   **{f"new_acc@{thr_str}": round(acc_new, 2)},
                   **({f"base_acc@{thr_str}": round(acc_base, 2)} if acc_base is not None else {}))

        # 仅对第一个评估（combined500）写 bridge（供可视化）
        if tag == "combined500":
            _write_bridge_artifacts(y_true=y, y_pred=pred_new)

def main():
    stop_probe = start_probe("offline")  # 资源采样
    t0 = time.time()

    try:
        print(f"[offline] start — dataset={OFFLINE_KEY} TRAIN_TRIGGER={TRAIN_TRIGGER}", flush=True)

        # 特征工程 & 划分 & 标准化（与原逻辑一致）
        Xtr, Xva, Ytr, Yva, SELECTED_FEATS, scaler = prepare_dataset(topk=10, val_frac=0.2, seed=42)

        # 训练前，如“已有 baseline”，先做一次 pre-eval（用于对比）
        baseline_existing = _try_load_existing_baseline()
        if baseline_existing is not None:
            Xva_s = scaler.transform(Xva.astype(np.float32))
            with torch.no_grad():
                pred = baseline_existing(torch.from_numpy(Xva_s).float().to(device)).cpu().numpy().ravel()
            pre_acc = calculate_accuracy_within_threshold(Yva, pred, ACC_THR)
            log_metric(component="offline", event="pre_eval", eval_acc=round(pre_acc, 2), eval_rows=int(len(Yva)))
            print(f"[offline] pre-eval accuracy={pre_acc:.2f}%", flush=True)

        # 训练（保持 FAST/FULL/早停 等原逻辑）
        do_train = TRAIN_TRIGGER != 0
        fast_mode = False
        model: nn.Module

        if not do_train and baseline_existing is None:
            do_train = True; fast_mode = True
            print("[offline] no baseline available; switching to FAST training mode.", flush=True)

        if do_train:
            max_epoch = FAST_MAX_EPOCH if fast_mode else FULL_MAX_EPOCH
            patience  = FAST_PATIENCE  if fast_mode else FULL_PATIENCE
            lr        = FAST_LR        if fast_mode else FULL_LR
            bs        = FAST_BS        if fast_mode else FULL_BS

            print(f"[offline] training start | mode={'FAST' if fast_mode else 'FULL'} | max_epoch={max_epoch} patience={patience} lr={lr} bs={bs}", flush=True)

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
                    xb = xb.to(device).float(); yb = yb.to(device).float().view(-1,1)
                    opt.zero_grad(set_to_none=True); lossf(model(xb), yb).backward(); opt.step()

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
                    print(f"[offline] early-stop @ epoch {ep} (no_improve={no_imp})", flush=True); break

            if best_state is not None:
                model = torch.load(best_state, map_location=device)
            print("[offline] training finished.", flush=True)
        else:
            # 不训练则沿用已有 baseline
            model = baseline_existing
            print("[offline] TRAIN_TRIGGER=0 and baseline exists → skip training.", flush=True)

        # 保存 baseline/model & latest（原逻辑）
        _save_models_and_latest(model)

        # ===== 新增：两套数据源 × 新/旧模型的双评估 =====
        _multi_eval(model_new=model, model_base=baseline_existing, scaler=scaler, feats=SELECTED_FEATS)

        # 记录离线阶段总耗时
        wall_s = round(time.time() - t0, 3)
        log_metric(component="offline", event="train_done",
                   mode=("FAST" if do_train and fast_mode else ("FULL" if do_train else "SKIP")),
                   wall_s=wall_s)

        os.makedirs("/tmp/kfp_outputs", exist_ok=True)
        open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
        print("[offline] done.", flush=True)
    finally:
        stop_probe()  # 上传 results/offline_resources.csv

if __name__ == "__main__":
    main()
