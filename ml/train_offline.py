#!/usr/bin/env python3

TRIGGER = 1  # 1 = 训练；0 = 跳过

import sys
import os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
import torch                          
import torch.nn as nn 
from torch.optim                import Adam
from torch.optim.lr_scheduler   import ReduceLROnPlateau
from torch.utils.data           import TensorDataset, DataLoader
from sklearn.preprocessing      import StandardScaler
from sklearn.decomposition      import PCA
from sklearn.model_selection    import train_test_split

from shared.minio_helper import load_csv, save_np, save_bytes
from shared.config       import MODEL_DIR, RESULT_DIR, TARGET_COL
from shared.metric_logger import log_metric
from shared.utils         import calculate_accuracy_within_threshold
from shared.features import FEATURE_COLS


from datetime import datetime

# ─── 0. 跳过训练逻辑 ──────────────────────────────────────────────
if TRIGGER == 0:
    print("[offline] TRIGGER=0 → 跳过训练，下载已有 artefacts")
    os.makedirs("/mnt/pvc/models",        exist_ok=True)
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

    from shared.config       import BUCKET
    from shared.minio_helper import s3, load_np

    # 下载模型/预处理器
    for fname in ("scaler.pkl","pca.pkl","model.pt"):
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{fname}")["Body"].read()
        with open(f"/mnt/pvc/models/{fname}", "wb") as f: f.write(raw)

    # 下载桥接预测结果
    for arr in ("bridge_true.npy","bridge_pred.npy"):
        try:
            a = load_np(f"{RESULT_DIR}/{arr}")
            np.save(f"/mnt/pvc/{RESULT_DIR}/{arr}", a)
        except: pass

    log_metric(component="offline", event="skip_train")
    sys.exit(0)

# ─── 1. 读取 & 清洗 ──────────────────────────────────────────────
df_bridge = (load_csv("datasets_old/old_total.csv")
             .replace({'<not counted>': np.nan})
             .replace(r'^\s*$', np.nan, regex=True)
             .dropna())
df_linear = (load_csv("datasets_old/old_linear.csv")
             .replace({'<not counted>': np.nan})
             .replace(r'^\s*$', np.nan, regex=True)
             .dropna())
df_dag1   = (load_csv("datasets_old/old_dag-1.csv")
             .replace({'<not counted>': np.nan})
             .replace(r'^\s*$', np.nan, regex=True)
             .dropna())

print(f"[offline] bridge rows={len(df_bridge)}  linear rows={len(df_linear)}  dag1 rows={len(df_dag1)}")

# —— 只丢掉 input_rate / latency，保留 output_rate 取 label ——  
for _df in (df_bridge, df_linear, df_dag1):
    _df.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
    _df.reindex(columns=FEATURE_COLS, fill_value=0.0, copy=False)

Xb = df_bridge[FEATURE_COLS].astype(np.float32).values
yb = df_bridge[TARGET_COL].astype(np.float32).values
Xl = df_linear[FEATURE_COLS].astype(np.float32).values
X1 = df_dag1 [FEATURE_COLS].astype(np.float32).values
y1 = df_dag1 [TARGET_COL].astype(np.float32).values


# ─── 2. 标准化 ──────────────────────────────────────────────────
scaler = StandardScaler().fit(Xb)
Xb_s   = scaler.transform(Xb)
Xl_s   = scaler.transform(Xl)
X1_s   = scaler.transform(X1)

# ─── 3. PCA（桥接上 fit，保留 ≥85% 方差） ────────────────────────
pca_full = PCA().fit(Xb_s)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n_comp   = int(np.searchsorted(cumvar, 0.85) + 1)
pca      = PCA(n_components=n_comp).fit(Xb_s)

Xb_p = pca.transform(Xb_s).astype(np.float32)
X1_p = pca.transform(X1_s).astype(np.float32)

# ─── 4. Train/Test Split on Bridge ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    Xb_p, yb, test_size=0.3, random_state=0
)

# ─── 5. DataLoader & 模型定义 ─────────────────────────────────────
bs = 16
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 32), nn.ReLU(),
            nn.Linear(32, 16),  nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

model  = MLP(n_comp).to(device)
opt    = Adam(model.parameters(), lr=1e-2)
sched  = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
lossfn = nn.SmoothL1Loss()

# ─── 6. 训练循环 (EarlyStopping patience=10) ─────────────────────
best_val, best_state, no_imp = float("inf"), None, 0
PATIENCE = 10
for epoch in range(1, 101):
    model.train()
    for xb_batch, yb_batch in train_loader:
        xb_batch = xb_batch.to(device)
        yb_batch = yb_batch.to(device)
        opt.zero_grad()
        l = lossfn(model(xb_batch), yb_batch)
        l.backward()
        opt.step()

    # 验证：直接用 train split 的 test 子集
    model.eval()
    with torch.no_grad():
        xb_t = torch.from_numpy(X_test).to(device)
        yb_t = torch.from_numpy(y_test).to(device)
        val_loss = float(lossfn(model(xb_t), yb_t).item())

    sched.step(val_loss)
    if val_loss + 1e-6 < best_val:
        best_val, best_state, no_imp = val_loss, model.state_dict(), 0
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"[offline] early-stop @ epoch {epoch}")
            break

print(f"[offline] best bridge val_loss = {best_val:.4f}")

# ─── 7. 预测 & 精度评估 ─────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    # 桥接拆分测试
    xb_t = torch.from_numpy(X_test).to(device)
    y_pred_bridge = model(xb_t).cpu().numpy()
    # 全量 dag1
    x1_t = torch.from_numpy(X1_p).to(device)
    y_pred_dag1   = model(x1_t).cpu().numpy()

acc_bridge = calculate_accuracy_within_threshold(y_test,       y_pred_bridge, 0.15)
acc_dag1   = calculate_accuracy_within_threshold(y1,           y_pred_dag1,   0.15)

print(f"[offline] bridge accuracy (thr=0.15) = {acc_bridge:.2f}%")
print(f"[offline] dag1   accuracy (thr=0.15) = {acc_dag1:.2f}%")

# ─── 8. 保存 artefacts & 上传 MinIO ───────────────────────────────
os.makedirs("/mnt/pvc/models",      exist_ok=True)
os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

joblib.dump(scaler, "/mnt/pvc/models/scaler.pkl")
joblib.dump(pca,    "/mnt/pvc/models/pca.pkl")
torch.save(best_state, "/mnt/pvc/models/model.pt")

np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy",  y_test)            # 模型拆分测试
np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy",  y_pred_bridge)

save_bytes(f"{MODEL_DIR}/scaler.pkl", open("/mnt/pvc/models/scaler.pkl","rb").read())
save_bytes(f"{MODEL_DIR}/pca.pkl", open("/mnt/pvc/models/pca.pkl","rb").read())      
save_bytes(f"{MODEL_DIR}/model.pt",
           open("/mnt/pvc/models/model.pt","rb").read())          # 初始 = 最新
save_bytes(f"{MODEL_DIR}/baseline_model.pt",
           open("/mnt/pvc/models/model.pt","rb").read())          # 永久基线

# 保存 config（隐藏层结构 / 激活）
cfg = {"hidden_layers": (32, 16), "activation": "relu"}
save_bytes(f"{MODEL_DIR}/last_model_config.json",
           json.dumps(cfg).encode(), "application/json")          # 对应 model.pt
save_bytes(f"{MODEL_DIR}/baseline_model_config.json",
           json.dumps(cfg).encode(), "application/json")          # 对应 baseline

# 更新时间戳
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           datetime.utcnow().isoformat() .encode())

# 推 result arrays
save_np(f"{RESULT_DIR}/bridge_true.npy" , y_test)
save_np(f"{RESULT_DIR}/bridge_pred.npy" , y_pred_bridge)

log_metric(component="offline", event="bridge_accuracy",accuracy=round(acc_bridge, 2))
log_metric(component="offline", event="dag1_accuracy",accuracy=round(acc_dag1, 2))

print("[offline] artefacts + baseline_model.pt 已上传 ✔")


# ─── 9. KFP V2 metadata 占位 ─────────────────────────────────────
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)

print("[offline] ✅ artefacts 写入 & 上传完毕")
