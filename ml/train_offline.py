#!/usr/bin/env python3

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
TMP_DIR      = "/tmp/offline_models"
os.makedirs(TMP_DIR, exist_ok=True)

device       = "cuda" if torch.cuda.is_available() else "cpu"
BRIDGE_N     = 500                 # Phase-1 桥接样本数
RAND_N       = 500                 # random_rates 测试子集大小

offline_path = "datasets/combined.csv"
test_path   = "datasets/random_rates.csv"

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
# 正常训练流程
# ---------------------------------------------------------------------
print("[offline] start full training")

# 1) 加载并清洗数据集
df_all = _read_clean(offline_path)

# ---------------------------------------------------------------------
# 2) 选取与输出最相关的前 10 个特征
# ---------------------------------------------------------------------
corrs = df_all[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL][FEATURE_COLS]
SELECTED_FEATS = corrs.abs().sort_values(ascending=False).head(10).index.tolist()
print(f"[offline] selected top10 feats: {SELECTED_FEATS}")

# 保存到 MinIO，供后续 dynamic_retrain 和 inference_consumer 加载
save_bytes(
    f"{MODEL_DIR}/selected_feats.json",
    json.dumps(SELECTED_FEATS).encode(),
    "application/json"
)

# 3) StandardScaler
# 3.1 准备训练数据（只用 SELECTED_FEATS 这 10 维）
X_full = df_all[SELECTED_FEATS].astype(np.float32).values
y_full = df_all[TARGET_COL].astype(np.float32).values

# 3.2 训练 scaler
scaler = StandardScaler().fit(X_full)
X_scaled = scaler.transform(X_full)

# 3.3 本地保存并上传 scaler
local_scaler = f"{TMP_DIR}/scaler.pkl"
joblib.dump(scaler, local_scaler)
with open(local_scaler, "rb") as f:
    save_bytes(f"{MODEL_DIR}/scaler.pkl", f.read(), "application/octet-stream")

# 3.4 记录输入维度
input_dim = X_scaled.shape[1]  # 应为 10

# ---------------------------------------------------------------------
# 4) 划分 / 训练模型
# ---------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_full, test_size=0.3, random_state=0
)
loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
    batch_size=16, shuffle=True
)

model = build_model(
    {"hidden_layers": (64, 32), "activation": "relu"},
    input_dim=input_dim
).to(device)
opt   = Adam(model.parameters(), lr=1e-2)
sched = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
lossf = nn.SmoothL1Loss()

best_val, best_state, no_imp = float("inf"), None, 0
for epoch in range(1, 101):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        lossf(model(xb), yb).backward()
        opt.step()
    with torch.no_grad():
        val = lossf(
            model(torch.from_numpy(X_te).to(device)),
            torch.from_numpy(y_te).to(device)
        ).item()
    sched.step(val)
    if val + 1e-6 < best_val:
        best_val, best_state, no_imp = val, model.state_dict(), 0
    else:
        no_imp += 1
        if no_imp >= 10:
            print(f"[offline] early-stop @ epoch {epoch}")
            break
model.load_state_dict(best_state)

# ---------------------------------------------------------------------
# 5) 精度测试
# ---------------------------------------------------------------------
# 5.1 bridge 子集 (combined.csv 尾 BRIDGE_N 条)
bridge_df = df_all.tail(BRIDGE_N).reset_index(drop=True)
Xb = scaler.transform(bridge_df[SELECTED_FEATS].values.astype(np.float32))
yb = bridge_df[TARGET_COL].astype(np.float32).values
with torch.no_grad():
    ypb = model(torch.from_numpy(Xb).to(device)).cpu().numpy()
acc_bridge = calculate_accuracy_within_threshold(yb, ypb, 0.15)

# 5.2 random_rates 子集
df_rand = _read_clean(test_path)
rand_sub = df_rand.head(min(RAND_N, len(df_rand))).reset_index(drop=True)
Xr = scaler.transform(rand_sub[SELECTED_FEATS].values.astype(np.float32))
yr = rand_sub[TARGET_COL].astype(np.float32).values
with torch.no_grad():
    ypr = model(torch.from_numpy(Xr).to(device)).cpu().numpy()
acc_rand = calculate_accuracy_within_threshold(yr, ypr, 0.15)

print(f"[offline] bridge({BRIDGE_N}) accuracy = {acc_bridge:.2f}%")
print(f"[offline] random({RAND_N}) accuracy = {acc_rand:.2f}%")
log_metric(component="offline", event="bridge_accuracy", accuracy=round(acc_bridge, 2))
log_metric(component="offline", event="random_accuracy", accuracy=round(acc_rand, 2))

# 在模型本体记录随机子集准确率
model._val_acc15 = float(acc_rand)

# ---------------------------------------------------------------------
# 6) 保存 artefacts & 上传 MinIO
# ---------------------------------------------------------------------
local_base  = f"{TMP_DIR}/baseline_model.pt"
local_model = f"{TMP_DIR}/model.pt"
torch.save(model.eval().cpu(), local_base)
shutil.copy(local_base, local_model)

for local_file, key in [
    (local_scaler,         f"{MODEL_DIR}/scaler.pkl"),
    (local_base,           f"{MODEL_DIR}/baseline_model.pt"),
    (local_model,          f"{MODEL_DIR}/model.pt"),
]:
    with open(local_file, "rb") as f:
        save_bytes(key, f.read(), "application/octet-stream")
print("[offline] artefacts uploaded ✔")

# ---------------------------------------------------------------------
# 7) 生成 bridge_true / bridge_pred（后续 plot 使用）
# ---------------------------------------------------------------------
for fname, arr in [("bridge_true.npy", yb), ("bridge_pred.npy", ypb)]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    save_bytes(f"{RESULT_DIR}/{fname}", buf.read(), "application/npy")
print("[offline] bridge artefacts uploaded ✔")

# ---------------------------------------------------------------------
# 8) Timestamp & metadata 占位
# ---------------------------------------------------------------------
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           datetime.utcnow().isoformat().encode(),
           "text/plain")
log_metric(component="offline", event="train_done")
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[offline] done.")