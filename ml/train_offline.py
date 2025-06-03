#!/usr/bin/env python3
"""
ml.train_offline – baseline 训练 + 产出基准预测 + 指标
"""
import os, datetime, io, time, json
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# -------- 项目内部工具 --------
from shared.minio_helper import load_csv, save_np, save_bytes        # ✅ 这里不再引入 save_json
from shared.utils        import save_json                           # ✅ 改为从 utils 导入
from shared.config       import DATA_DIR, MODEL_DIR, RESULT_DIR, TARGET_COL
from shared.features     import FEATURE_COLS
from shared.utils        import calculate_accuracy_within_threshold
from shared.metric_logger import log_metric

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)

class MLPBaseline(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64,  32),  nn.ReLU(),
            nn.Linear(32,   1),
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ---------- 1. 读取数据 ----------
df   = load_csv(f"{DATA_DIR}/combined.csv")
Xraw = df[FEATURE_COLS].astype(np.float32).values
y    = df[TARGET_COL].astype(np.float32).values

# ---------- 2. 训练 ----------
scaler = StandardScaler().fit(Xraw)
Xs     = scaler.transform(Xraw)
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MLPBaseline(Xs.shape[1]).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = nn.SmoothL1Loss()

Xts = torch.from_numpy(Xs).to(device)
yts = torch.from_numpy(y).to(device)

t0 = time.perf_counter()
for _ in range(200):
    model.train(); opt.zero_grad()
    loss = lossfn(model(Xts), yts); loss.backward(); opt.step()
train_time_s = round(time.perf_counter() - t0, 3)

# ---------- 3. 预测 & 指标 ----------
with torch.no_grad():
    pred = model(Xts).cpu().numpy().astype(np.float32)

mae  = float(mean_absolute_error(y, pred))
rmse = float(np.sqrt(mean_squared_error(y, pred)))
acc  = calculate_accuracy_within_threshold(y, pred, 0.15)

print(f"[offline] acc={acc:.2f}% | MAE={mae:.3f} | RMSE={rmse:.3f} | rows={len(y)}")

# ---------- 4. 保存 artefacts ----------
os.makedirs("/mnt/pvc/models", exist_ok=True)
os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

joblib.dump(scaler, "/mnt/pvc/models/scaler.pkl")
torch.save(model.state_dict(), "/mnt/pvc/models/model.pt")

np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy", y)
np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy", pred)

save_np(f"{RESULT_DIR}/bridge_true.npy", y)
save_np(f"{RESULT_DIR}/bridge_pred.npy", pred)

save_bytes(f"{MODEL_DIR}/scaler.pkl", open("/mnt/pvc/models/scaler.pkl","rb").read())
save_bytes(f"{MODEL_DIR}/model.pt"  , open("/mnt/pvc/models/model.pt" ,"rb").read())

# ---------- 5. 指标上报 ----------
model_size_mb = round(os.path.getsize("/mnt/pvc/models/model.pt") / 2**20, 3)
log_metric(
    component="offline",
    event="train_done",
    train_rows=len(y),
    train_time_s=train_time_s,
    model_size_mb=model_size_mb,
    accuracy=round(acc, 2),
    mae=round(mae, 4),
    rmse=round(rmse, 4),
)

# 旧 json（给别的脚本用）
meta = dict(
    component   ="offline_train",
    rows        =int(len(y)),
    baseline_acc=round(acc,2),
    mae=round(mae,4),
    rmse=round(rmse,4),
    utc_end     =datetime.datetime.utcnow().isoformat()+"Z",
    model_bytes =os.path.getsize("/mnt/pvc/models/model.pt"),
    train_time_s=train_time_s,
)
save_json(f"{RESULT_DIR}/timing/offline_train.json", meta)

print("[offline] artefacts & metrics uploaded")
