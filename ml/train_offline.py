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

TRIGGER = 1  # 1 = 训练；0 = 跳过

import sys, os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime

from shared.minio_helper import load_csv, load_np, save_np, save_bytes, s3
from shared.config import MODEL_DIR, RESULT_DIR, TARGET_COL, BUCKET, DATA_DIR
from shared.metric_logger import log_metric
from shared.features import FEATURE_COLS
from shared.utils import calculate_accuracy_within_threshold

if TRIGGER == 0:
    print("[offline] TRIGGER=0 → 跳过训练，下载已有 artefacts")
    os.makedirs("/mnt/pvc/models", exist_ok=True)
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

    for fname in ("scaler.pkl", "pca.pkl", "model.pt"):
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{fname}")["Body"].read()
        with open(f"/mnt/pvc/models/{fname}", "wb") as f: f.write(raw)

    for arr in ("bridge_true.npy", "bridge_pred.npy"):
        try:
            a = load_np(f"{RESULT_DIR}/{arr}")
            np.save(f"/mnt/pvc/{RESULT_DIR}/{arr}", a)
        except: pass

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

device = "cuda" if torch.cuda.is_available() else "cpu"

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
os.makedirs("/mnt/pvc/models",        exist_ok=True)
os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

# 1) 本地持久化
joblib.dump(scaler, "/mnt/pvc/models/scaler.pkl")
joblib.dump(pca,    "/mnt/pvc/models/pca.pkl")
torch.save(best_state, "/mnt/pvc/models/model.pt")

np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy", y_te)
np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy", y_pred)

# 2) 上传 MinIO
for fn in ("scaler.pkl", "pca.pkl", "model.pt"):
    with open(f"/mnt/pvc/models/{fn}", "rb") as f:
        save_bytes(f"{MODEL_DIR}/{fn}", f.read())

# ---------------- 新增：自动提取隐藏层宽度 ----------------
hidden_layers = tuple(
    l.out_features for l in model.net if isinstance(l, nn.Linear)
)[:-1]  # 去掉最后一层输出层
cfg = {"hidden_layers": hidden_layers, "activation": "relu"}

save_bytes(f"{MODEL_DIR}/last_model_config.json",
           json.dumps(cfg).encode(), "application/json")
save_bytes(f"{MODEL_DIR}/baseline_model_config.json",
           json.dumps(cfg).encode(), "application/json")
save_bytes(f"{MODEL_DIR}/baseline_model.pt",
           open("/mnt/pvc/models/model.pt", "rb").read())
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           datetime.utcnow().isoformat().encode())

# 3) 记录指标
log_metric(component="offline", event="test_accuracy",
           accuracy=round(acc, 2))
print("[offline] artefacts uploaded ✔")

# ---------- 9. KFP metadata ----------
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
