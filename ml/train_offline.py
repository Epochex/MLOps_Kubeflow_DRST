#!/usr/bin/env python3
"""
ml.train_offline.py – baseline training / skip with artefacts (MinIO‑only)
──────────────────────────────────────────────────────────────────────────────
* TRIGGER=1  → 重新训练 (datasets/combined.csv) 并更新 artefacts
* TRIGGER=0  → 跳过训练；下载已有 artefacts，
               将 baseline_model.pt 拷贝为 model.pt，
               并重新生成 bridge_true.npy / bridge_pred.npy
* Phase‑1 桥接基线 = combined.csv **最后 500 条**
* 精度测试 (两种子集各 500 条) 打印 & 记录 metric_logger
* 完全移除 PVC：所有中间文件写 /tmp/offline_models，然后立即上传 MinIO
"""

from __future__ import annotations
import os, io, sys, json, shutil, random, joblib
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from shared.minio_helper import load_csv, save_bytes, s3, BUCKET
from shared.config       import MODEL_DIR, RESULT_DIR, TARGET_COL, DATA_DIR
from shared.metric_logger import log_metric
from shared.features     import FEATURE_COLS
from shared.utils        import calculate_accuracy_within_threshold
from ml.model            import build_model

# ---------------------------------------------------------------------
# 常量 & 目录
# ---------------------------------------------------------------------
TRIGGER   = int(os.getenv("TRIGGER", "1"))     # 1 = 训练；0 = 跳过
TMP_DIR   = "/tmp/offline_models"
os.makedirs(TMP_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
BRIDGE_N = 500                 # Phase‑1 样本数
RAND_N   = 500                 # random_rates 测试子集大小

combined_path = f"datasets/combined.csv"
random_path   = "datasets/random_rates.csv"   # 相对路径即可

# ---------------------------------------------------------------------
# 共用的预处理函数
# ---------------------------------------------------------------------

def _read_clean(path: str) -> pd.DataFrame:
    """读取、替换空值、去掉无关列，并按 FEATURE_COLS+TARGET_COL 排列"""
    df = (load_csv(path)
          .replace({"<not counted>": np.nan})
          .replace(r"^\s*$", np.nan, regex=True)
          .dropna())
    df.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
    return df.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

# ---------------------------------------------------------------------
# branch‑0: 跳过训练，直接拷 artefacts 并重新生成 bridge_true / bridge_pred
# ---------------------------------------------------------------------
if TRIGGER == 0:
    print("[offline][SKIP] TRIGGER=0 → skip training, reuse artefacts")

    local_scaler = f"{TMP_DIR}/scaler.pkl"
    local_pca    = f"{TMP_DIR}/pca.pkl"
    local_base   = f"{TMP_DIR}/baseline_model.pt"
    local_model  = f"{TMP_DIR}/model.pt"

    # 1) 从 MinIO 下载 artefacts
    for key, local in [
        (f"{MODEL_DIR}/scaler.pkl", local_scaler),
        (f"{MODEL_DIR}/pca.pkl"   , local_pca   ),
        (f"{MODEL_DIR}/baseline_model.pt", local_base)]:
        print(f"[offline][SKIP] download {key}")
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        with open(local, "wb") as f: f.write(raw)

    shutil.copy(local_base, local_model)  # baseline → model
    with open(local_model, "rb") as f:
        save_bytes(f"{MODEL_DIR}/model.pt", f.read(), "application/octet-stream")
    print("[offline][SKIP] model.pt overwritten with baseline clone")

    # 2) 重新生成 bridge_true / bridge_pred 供后续 Plot 使用
    df_all   = _read_clean(combined_path)
    bridge_df = df_all.tail(BRIDGE_N).reset_index(drop=True)

    scaler = joblib.load(local_scaler)
    pca    = joblib.load(local_pca)
    model  = torch.load(local_base, map_location=device).eval()

    #  跳过训练后下载的模型精度测试
    with torch.no_grad():
        # bridge subset
        Xb = pca.transform(scaler.transform(bridge_df[FEATURE_COLS].values)).astype(np.float32)
        yb = bridge_df[TARGET_COL].astype(np.float32).values
        ypb = model(torch.from_numpy(Xb).to(device)).cpu().numpy()
        acc_bridge = calculate_accuracy_within_threshold(yb, ypb, 0.15)

        # random_rates subset
        rand_sub = df_rand.sample(n=min(RAND_N, len(df_rand)), random_state=0).reset_index(drop=True)
        Xr = pca.transform(scaler.transform(rand_sub[FEATURE_COLS].values)).astype(np.float32)
        yr = rand_sub[TARGET_COL].astype(np.float32).values
        ypr = model(torch.from_numpy(Xr).to(device)).cpu().numpy()
        acc_rand = calculate_accuracy_within_threshold(yr, ypr, 0.15)

    print(f"[offline] bridge(500) accuracy = {acc_bridge:.2f}%")
    print(f"[offline] random(500) accuracy = {acc_rand:.2f}%")
    log_metric(component="offline", event="bridge_accuracy", accuracy=round(acc_bridge, 2))
    log_metric(component="offline", event="random_accuracy", accuracy=round(acc_rand, 2))
    
    # KFP metadata 占位
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
    sys.exit(0)

# ---------------------------------------------------------------------
# branch‑1: 正常训练
# ---------------------------------------------------------------------
print("[offline] TRIGGER=1 → start full training")

# 1) 加载数据集

df_all   = _read_clean(combined_path)
df_rand  = _read_clean(random_path)
bridge_df = df_all.tail(BRIDGE_N).reset_index(drop=True)

print(f"[offline] combined={len(df_all)}  random_rates={len(df_rand)}  bridge={len(bridge_df)}")

X_full = df_all[FEATURE_COLS].astype(np.float32).values
y_full = df_all[TARGET_COL].astype(np.float32).values

# 2) Scaler + PCA
scaler = StandardScaler().fit(X_full)
X_scaled = scaler.transform(X_full)

cumvar = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.85) + 1)
pca    = PCA(n_components=n_comp).fit(X_scaled)
X_pca  = pca.transform(X_scaled).astype(np.float32)

# 3) 训练模型
X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y_full, test_size=0.3, random_state=0)
loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=16, shuffle=True)

model = build_model({"hidden_layers": (64, 32), "activation": "relu"}, input_dim=n_comp).to(device)
opt   = Adam(model.parameters(), lr=1e-2)
sched = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
lossf = nn.SmoothL1Loss()

best_val, best_state, no_imp = float("inf"), None, 0
for epoch in range(1, 101):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    with torch.no_grad():
        val = lossf(model(torch.from_numpy(X_te).to(device)), torch.from_numpy(y_te).to(device)).item()
    sched.step(val)
    if val + 1e-6 < best_val:
        best_val, best_state, no_imp = val, model.state_dict(), 0
    else:
        no_imp += 1
        if no_imp >= 10:
            print(f"[offline] early-stop @ epoch {epoch}"); break
model.load_state_dict(best_state)

# 4) 精度测试
with torch.no_grad():
    # bridge subset
    Xb = pca.transform(scaler.transform(bridge_df[FEATURE_COLS].values)).astype(np.float32)
    yb = bridge_df[TARGET_COL].astype(np.float32).values
    ypb = model(torch.from_numpy(Xb).to(device)).cpu().numpy()
    acc_bridge = calculate_accuracy_within_threshold(yb, ypb, 0.15)

    # random_rates subset
    rand_sub = df_rand.sample(n=min(RAND_N, len(df_rand)), random_state=0).reset_index(drop=True)
    Xr = pca.transform(scaler.transform(rand_sub[FEATURE_COLS].values)).astype(np.float32)
    yr = rand_sub[TARGET_COL].astype(np.float32).values
    ypr = model(torch.from_numpy(Xr).to(device)).cpu().numpy()
    acc_rand = calculate_accuracy_within_threshold(yr, ypr, 0.15)

print(f"[offline] bridge(500) accuracy = {acc_bridge:.2f}%")
print(f"[offline] random(500) accuracy = {acc_rand:.2f}%")
log_metric(component="offline", event="bridge_accuracy", accuracy=round(acc_bridge, 2))
log_metric(component="offline", event="random_accuracy", accuracy=round(acc_rand, 2))

# ---------- 4. 评估完成后，把 acc 写进模型本体 ----------
# 这里用 random_rates 子集的 acc 作为“通用基准”
model._val_acc15 = float(acc_rand)  

# 5) 保存 artefacts & 上传 MinIO
local_scaler = f"{TMP_DIR}/scaler.pkl"
local_pca    = f"{TMP_DIR}/pca.pkl"
local_base   = f"{TMP_DIR}/baseline_model.pt"
local_model  = f"{TMP_DIR}/model.pt"

joblib.dump(scaler, local_scaler)
joblib.dump(pca,    local_pca)
torch.save(model.eval().cpu(), local_base)
shutil.copy(local_base, local_model)

for local, key in [
    (local_scaler, f"{MODEL_DIR}/scaler.pkl"),
    (local_pca,    f"{MODEL_DIR}/pca.pkl"),
    (local_base,   f"{MODEL_DIR}/baseline_model.pt"),
    (local_model,  f"{MODEL_DIR}/model.pt")]:
    with open(local, "rb") as f:
        save_bytes(key, f.read(), "application/octet-stream")
print("[offline] artefacts uploaded ✔")

# 6) 保存 bridge_true / bridge_pred
for fname, arr in [("bridge_true.npy", yb), ("bridge_pred.npy", ypb)]:
    buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
    save_bytes(f"{RESULT_DIR}/{fname}", buf.read(), "application/npy")
print("[offline] bridge artefacts uploaded ✔")

# 7) timestamp & metadata
save_bytes(f"{MODEL_DIR}/last_update_utc.txt", datetime.utcnow().isoformat().encode(), "text/plain")
log_metric(component="offline", event="train_done")

os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[offline] done.")
