#!/usr/bin/env python3
"""
ml.train_offline.py - baseline MLP with PCA (bridge only)
────────────────────────────────────────────────────────────
• 若 TRIGGER = 0，跳过训练并下载已有 artefacts
• 若 TRIGGER = 1，从 datasets_old/old_total.csv 训练
• 仅使用 bridge 行数据
• PCA 保留 ≥85% 方差
• 输出 scaler / pca / model.pt / 预测结果等
"""

TRIGGER = 0  # 1 = 训练；0 = 跳过

import sys, os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict, Tuple

from shared.minio_helper import load_csv, load_np, save_np, save_bytes, s3
from shared.config import MODEL_DIR, RESULT_DIR, TARGET_COL, BUCKET, DATA_DIR
from shared.metric_logger import log_metric
from shared.features import FEATURE_COLS
from shared.utils import calculate_accuracy_within_threshold, _bytes_to_model

device = "cuda" if torch.cuda.is_available() else "cpu"

if TRIGGER == 0:
    print("[offline] TRIGGER=0 → 跳过训练，下载已有 artefacts")
    os.makedirs("/mnt/pvc/models", exist_ok=True)
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

    # 1) 下载预处理器 + 两个模型
    for fname in ("scaler.pkl", "pca.pkl", "baseline_model.pt", "model.pt"):
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{fname}")["Body"].read()
        local = f"/mnt/pvc/models/{fname}"
        with open(local, "wb") as f:
            f.write(raw)
        print(f"[offline] downloaded {fname} → {local}")

    # 2) 加载
    scaler = joblib.load("/mnt/pvc/models/scaler.pkl")
    pca    = joblib.load("/mnt/pvc/models/pca.pkl")

    with open("/mnt/pvc/models/baseline_model.pt", "rb") as f:
        baseline_model = _bytes_to_model(f.read())       # ✔ 兼容 state_dict / 完整模型

    with open("/mnt/pvc/models/model.pt", "rb") as f:
        adaptive_model = _bytes_to_model(f.read())       # 同理

    # 3) 对齐函数（给 adaptive model 用）
    def _align_adaptive_input(X_scaled: np.ndarray, model: nn.Module) -> np.ndarray:
        in_dim = model.net[0].in_features
        if X_scaled.shape[1] >= in_dim:
            return X_scaled[:, :in_dim]
        pad = np.zeros((X_scaled.shape[0], in_dim - X_scaled.shape[1]), dtype=np.float32)
        return np.concatenate([X_scaled, pad], axis=1)

    # —— 在 old_total.csv 上测两条线路的精度 ——  
    df_tot = (load_csv(f"{DATA_DIR}/old_total.csv")
              .replace({'<not counted>': np.nan})
              .dropna())
    df_tot.drop(columns=["input_rate","latency"], errors="ignore", inplace=True)
    df_tot = df_tot.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

    X_raw      = df_tot[FEATURE_COLS].astype(np.float32).values
    y_true_tot = df_tot[TARGET_COL].astype(np.float32).values

    # 3.1 PCA → baseline
    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled).astype(np.float32)
    with torch.no_grad():
        y_base = baseline_model(torch.from_numpy(X_pca).to(device)).cpu().numpy()
    acc_base_tot = calculate_accuracy_within_threshold(y_true_tot, y_base, 0.15)
    print(f"[offline][SKIP] baseline@old_total acc = {acc_base_tot:.2f}%")
    log_metric(component="offline", event="baseline_old_total_acc",
               accuracy=round(acc_base_tot,2))

    # 3.2 Scaler → adaptive
    X_adpt = _align_adaptive_input(X_scaled, adaptive_model)
    with torch.no_grad():
        y_adpt = adaptive_model(torch.from_numpy(X_adpt).to(device)).cpu().numpy()
    acc_adpt_tot = calculate_accuracy_within_threshold(y_true_tot, y_adpt, 0.15)
    print(f"[offline][SKIP] adaptive@old_total acc = {acc_adpt_tot:.2f}%")
    log_metric(component="offline", event="adaptive_old_total_acc",
               accuracy=round(acc_adpt_tot,2))

    # —— 在 old_dag-1.csv 上同样测试 ——  
    try:
        df_d1 = (load_csv(f"{DATA_DIR}/old_dag-1.csv")
                 .replace({'<not counted>': np.nan})
                 .dropna())
        df_d1.drop(columns=["input_rate","latency"], errors="ignore", inplace=True)
        df_d1 = df_d1.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

        X1_raw      = df_d1[FEATURE_COLS].astype(np.float32).values
        y_true_d1   = df_d1[TARGET_COL].astype(np.float32).values
        X1_scaled   = scaler.transform(X1_raw)

        # baseline
        X1_pca      = pca.transform(X1_scaled).astype(np.float32)
        with torch.no_grad():
            y1_base = baseline_model(torch.from_numpy(X1_pca).to(device)).cpu().numpy()
        acc1_base = calculate_accuracy_within_threshold(y_true_d1, y1_base, 0.15)
        print(f"[offline][SKIP] baseline@old_dag1 acc = {acc1_base:.2f}%")
        log_metric(component="offline", event="baseline_dag1_acc",
                   accuracy=round(acc1_base,2))

        # adaptive
        X1_adpt    = _align_adaptive_input(X1_scaled, adaptive_model)
        with torch.no_grad():
            y1_adpt = adaptive_model(torch.from_numpy(X1_adpt).to(device)).cpu().numpy()
        acc1_adpt = calculate_accuracy_within_threshold(y_true_d1, y1_adpt, 0.15)
        print(f"[offline][SKIP] adaptive@old_dag1 acc = {acc1_adpt:.2f}%")
        log_metric(component="offline", event="adaptive_dag1_acc",
                   accuracy=round(acc1_adpt,2))

    except Exception as e:
        print(f"[offline] old_dag-1 accuracy calc failed: {e}")

    # 最后打个 skip_train
    log_metric(component="offline", event="skip_train")
    sys.exit(0)

# ---------- 1. 读取 old_total.csv ----------
df = (load_csv("{DATA_DIR}/old_total.csv")
      .replace({'<not counted>': np.nan})
      .dropna())
df.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
df = df.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

X = df[FEATURE_COLS].astype(np.float32).values
y = df[TARGET_COL].astype(np.float32).values

# ---------- 2. 标准化 ----------
scaler = StandardScaler().fit(X)
Xs     = scaler.transform(X)

# ---------- 3. PCA (保留至85% 方差) ----------
cumvar = np.cumsum(PCA().fit(Xs).explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.85) + 1)
pca    = PCA(n_components=n_comp).fit(Xs)
Xp     = pca.transform(Xs).astype(np.float32)

# ---------- 4. Train / Test Split ----------
X_tr, X_te, y_tr, y_te = train_test_split(Xp, y, test_size=0.3, random_state=0)

# ---------- 5. DataLoader & 模型 ----------
bs = 16
tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                     torch.from_numpy(y_tr)), batch_size=bs, shuffle=True)


class MLP(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16,  1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

model  = MLP(n_comp).to(device)
opt    = Adam(model.parameters(), lr=1e-2)
sched  = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
lossf  = nn.SmoothL1Loss()

# ---------- 6. 训练（Early-Stop） ----------
best_val, best_state, no_imp = float("inf"), None, 0
for epoch in range(1, 101):
    model.train()
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = float(lossf(model(torch.from_numpy(X_te).to(device)),
                               torch.from_numpy(y_te).to(device)).item())
    sched.step(val_loss)
    if val_loss + 1e-6 < best_val:
        best_val, best_state, no_imp = val_loss, model.state_dict(), 0
    else:
        no_imp += 1
        if no_imp >= 10:
            print(f"[offline] early-stop @ epoch {epoch}")
            break

model.load_state_dict(best_state)

# ---------- 7. 评估 ----------
with torch.no_grad():
    y_pred = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
acc = calculate_accuracy_within_threshold(y_te, y_pred, 0.15)
print(f"[offline] test accuracy = {acc:.2f}%  (thr 0.15)")

try:
    df_d1 = (load_csv(f"{DATA_DIR}/old_dag-1.csv")
             .replace({'<not counted>': np.nan})
             .dropna())
    df_d1.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
    # 保证和训练时用的一样的列顺序
    df_d1 = df_d1.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

    # 特征 / 标签
    X1 = df_d1[FEATURE_COLS].astype(np.float32).values
    y1 = df_d1[TARGET_COL].astype(np.float32).values

    # 标准化 + PCA
    X1s = scaler.transform(X1)
    X1p = pca.transform(X1s).astype(np.float32)

    # 预测 & 准确率
    with torch.no_grad():
        y1_pred = model(torch.from_numpy(X1p).to(device)).cpu().numpy()
    acc_d1 = calculate_accuracy_within_threshold(y1, y1_pred, 0.15)

    print(f"[offline] dag1 accuracy = {acc_d1:.2f}%  (thr 0.15)")
    log_metric(component="offline", event="dag1_accuracy", accuracy=round(acc_d1,2))
except Exception as e:
    print(f"[offline] dag1 accuracy calc failed: {e}")


# ---------- 8. 保存 artefacts ----------
import os
from datetime import datetime
from shared.minio_helper import save_bytes

# 确保路径存在
os.makedirs("/mnt/pvc/models",        exist_ok=True)
os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

# 1) 本地持久化：保存 scaler、pca，以及完整的模型对象
scaler_path = "/mnt/pvc/models/scaler.pkl"
pca_path    = "/mnt/pvc/models/pca.pkl"
model_path  = "/mnt/pvc/models/model.pt"
true_path   = f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy"
pred_path   = f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy"

# 保存预处理对象
joblib.dump(scaler, scaler_path)
joblib.dump(pca,    pca_path)

# ←—— 这里改为保存完整模型对象 ——→
# model 已经用 best_state load 好了：
model.load_state_dict(best_state)
torch.save(model.eval().cpu(), model_path)

# 保存测试集真值与预测
np.save(true_path, y_te)
np.save(pred_path, y_pred)

print(f"[offline] scaler saved to: {scaler_path}")
print(f"[offline] pca    saved to: {pca_path}")
print(f"[offline] model  saved to: {model_path}")
print(f"[offline] bridge_true.npy → {true_path}")
print(f"[offline] bridge_pred.npy → {pred_path}")

# 2) 上传到 MinIO
#   — scaler / pca / model.pt（完整模型）
for fn, local in [
    ("scaler.pkl", scaler_path),
    ("pca.pkl",    pca_path),
    ("model.pt",   model_path),
]:
    with open(local, "rb") as f:
        save_bytes(f"{MODEL_DIR}/{fn}", f.read(), "application/octet-stream")

# 3) 上传 bridge_true / bridge_pred
save_bytes(f"{RESULT_DIR}/bridge_true.npy",
           open(true_path, "rb").read(),
           "application/npy")
save_bytes(f"{RESULT_DIR}/bridge_pred.npy",
           open(pred_path, "rb").read(),
           "application/npy")

# 4) 上传配置（可选，保持原样）
hidden_layers = tuple(
    l.out_features for l in model.net if isinstance(l, nn.Linear)
)[:-1]
cfg = {"hidden_layers": hidden_layers, "activation": "relu"}
save_bytes(f"{MODEL_DIR}/baseline_model_config.json",
           json.dumps(cfg).encode(), "application/json")
save_bytes(f"{MODEL_DIR}/last_model_config.json",
           json.dumps(cfg).encode(), "application/json")

# 5) 上传时间戳
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           datetime.utcnow().isoformat().encode(), "text/plain")

print("[offline] artefacts uploaded ✔")

# ---------- 9. KFP metadata ----------
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
