#!/usr/bin/env python3
"""
ml.dynamic_retrain ─ JS 触发的在线增量重训练  
新的实现：**累积全部已见流数据** 而非仅滑动窗口
"""
import os
import sys, io, json, itertools, joblib, numpy as np, torch
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.features     import FEATURE_COLS
from ml.model            import build_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler     import Timer

# ---------- 1. 当前 JS ----------
JS = float(sys.argv[1])

# ---------- 2. 合并 “最新窗口” + “历史累计” ----------
latest_rows = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True).tolist()
cumu_path   = "/mnt/pvc/all_seen.npy"

if os.path.exists(cumu_path):
    all_seen = np.load(cumu_path, allow_pickle=True).tolist()
else:
    all_seen = []

def _row_key(r):
    feats = tuple(r["features"][c] for c in FEATURE_COLS)
    return feats + (r["label"],)

merged = { _row_key(r): r for r in all_seen }
merged.update({ _row_key(r): r for r in latest_rows })
all_rows = list(merged.values())

# 持久化新的累计数据，供下次 retrain 使用
np.save(cumu_path, np.array(all_rows, dtype=object))

# ---------- 3. 超参网格选择 ----------
param_grid_A = {
    "learning_rate": [1e-3],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32)],
    "activation":    ["relu"]
}
param_grid_B = {
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32), (128, 64, 32)],
    "activation":    ["relu"]
}
param_grid_C = {
    "learning_rate": [1e-2, 1e-3],
    "batch_size":    [8, 16],
    "hidden_layers": [(128, 64, 32)],
    "activation":    ["relu", "gelu"],
    "loss":          ["Huber", "mse"]
}
def _select_grid(js):
    if js <= JS_SEV1_THRESH / 3:   return None
    if js <= JS_SEV1_THRESH:       return param_grid_A
    if js <= JS_SEV2_THRESH:       return param_grid_B
    return param_grid_C

grid_cfg = _select_grid(JS)
if grid_cfg is None:
    print("[dynamic] JS below skip threshold → exit")
    sys.exit(0)

# ---------- 3. 载 scaler ----------
buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

# ---------- 4. 载最新窗口 ----------
batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)
Xr = np.array([[r["features"][c] for c in FEATURE_COLS] for r in batch], np.float32)
yr = np.array([r["label"] for r in batch], np.float32)

Xr = scaler.transform(Xr)
Xtr,Xval, ytr,yval = train_test_split(Xr, yr, test_size=0.2, random_state=0)

Xtr_t = torch.from_numpy(Xtr)
ytr_t = torch.from_numpy(ytr)
Xval_t= torch.from_numpy(Xval)
yval_t= torch.from_numpy(yval)

LOSS_MAP = {"Huber": nn.SmoothL1Loss, "mse": nn.MSELoss}
best_loss, best_state, best_cfg = float("inf"), None, None
device  = "cpu"

# ---------- 5. gird-search ----------
with Timer("Dynamic_Retrain", "retrain"):
    for hp in itertools.product(*grid_cfg.values()):
        cfg = dict(zip(grid_cfg.keys(), hp))
        model = build_model(cfg, Xtr.shape[1]).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        lossf = LOSS_MAP.get(cfg.get("loss","Huber"), nn.SmoothL1Loss)()

        for _ in range(10):                      # 10 mini-epochs
            opt.zero_grad()
            loss = lossf(model(Xtr_t), ytr_t)
            loss.backward(); opt.step()

        with torch.no_grad():
            val_loss = lossf(model(Xval_t), yval_t).item()

        if val_loss < best_loss:
            best_loss, best_state, best_cfg = val_loss, model.state_dict(), cfg
            print(f"[dynamic] ★ new best {best_loss:.4f} cfg={cfg}")

# ---------- 6. 推送 artefacts ----------
# 保存完整模型（而不是 state_dict）
best_model = build_model(best_cfg, Xtr.shape[1])
best_model.load_state_dict(best_state)

buf = io.BytesIO()
torch.save(best_model.eval().cpu(), buf)   # ① 直接 dump 模型对象
buf.seek(0)
save_bytes(f"{MODEL_DIR}/model.pt", buf.read())

# 把 config 一并保存，便于将来回溯
save_bytes(f"{MODEL_DIR}/last_model_config.json",
           json.dumps(best_cfg).encode(), "application/json")
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           (datetime.utcnow().isoformat() + "Z").encode())


# ←—— 这里是修改点 ——→
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           (datetime.utcnow().isoformat() + "Z").encode())

log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update", value=round(best_loss,6))
sync_all_metrics_to_minio()

open("/mnt/pvc/retrain_done.flag","w").close()          # ★ 调试用
print(f"[dynamic] done, best_loss={best_loss:.6f}, cfg={best_cfg}")
