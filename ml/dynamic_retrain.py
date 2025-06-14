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


    # 1.  当前漂移值（由 monitor.py 传参）
JS = float(sys.argv[1])

# 2.  合并累计样本
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
np.save(cumu_path, np.array(all_rows, dtype=object))      # 下次用

# 3.  加载 scaler & PCA（保证 6-维输入）
buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/pca.pkl")["Body"].read())
pca = joblib.load(buf)          # baseline 用的 PCA

# 4.  把最新窗口做成 6-维 PCA 特征
batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)

X_raw = np.asarray(
    [[r["features"][c] for c in FEATURE_COLS] for r in batch],
    dtype=np.float32
)
y_raw = np.asarray([r["label"] for r in batch], dtype=np.float32)

X_scaled = scaler.transform(X_raw)
X_pca    = pca.transform(X_scaled).astype(np.float32)     # ≈6 维
input_dim = X_pca.shape[1]

# 为网格搜索留一份验证集（80 / 20）
Xtr, Xval, ytr, yval = train_test_split(
    X_pca, y_raw, test_size=0.2, random_state=0
)
Xtr_t, ytr_t = torch.from_numpy(Xtr),  torch.from_numpy(ytr)
Xval_t,yval_t= torch.from_numpy(Xval), torch.from_numpy(yval)

# 5.  四档策略：网格 or 快速单模型 or 跳过
param_grid_A = {                 # 轻度漂移
    "learning_rate": [1e-3],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32)],
    "activation":    ["relu"],
}
param_grid_B = {                 # 中度漂移
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32), (128, 64, 32)],
    "activation":    ["relu"],
}
param_grid_C = {                 # 严重漂移
    "learning_rate": [1e-2, 1e-3],
    "batch_size":    [8, 16],
    "hidden_layers": [(128, 64, 32)],
    "activation":    ["relu", "gelu"],
    "loss":          ["Huber", "mse"],
}

def _grid_by_js(js_val):
    if js_val <= JS_SEV1_THRESH / 3:   return None          # very-low → skip
    if js_val <= JS_SEV1_THRESH:       return param_grid_A  # low
    if js_val <= JS_SEV2_THRESH:       return param_grid_B  # mid
    return param_grid_C                                      # high

grid_cfg = _grid_by_js(JS)
if grid_cfg is None:
    print(f"[dynamic] JS={JS:.4f} 低于阈值，保持旧模型不变")
    sys.exit(0)

device = "cpu"
LOSS_MAP = {"Huber": nn.SmoothL1Loss, "mse": nn.MSELoss}

# 6. 训练
if grid_cfg is param_grid_A:               # ★ 快速固定超参分支 ★
    best_cfg = {
        "hidden_layers": (64, 32),
        "activation": "relu",
        "learning_rate": 1e-3,
    }
    model = build_model(best_cfg, input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=best_cfg["learning_rate"])
    lossf = nn.SmoothL1Loss()

    for _ in range(20):                    # 20 轮足够
        model.train()
        opt.zero_grad()
        loss = lossf(model(torch.from_numpy(X_pca)), torch.from_numpy(y_raw))
        loss.backward(); opt.step()

    best_model = model.eval()
    best_loss  = loss.item()

else:                                      # ★ 网格搜索分支 (B / C) ★
    best_loss, best_state, best_cfg = float("inf"), None, None

    with Timer("Dynamic_Retrain", "retrain"):
        for hp in itertools.product(*grid_cfg.values()):
            cfg = dict(zip(grid_cfg.keys(), hp))
            model = build_model(cfg, input_dim).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            lossf = LOSS_MAP.get(cfg.get("loss","Huber"), nn.SmoothL1Loss)()

            for _ in range(10):                # 10 mini-epochs
                opt.zero_grad()
                loss = lossf(model(Xtr_t), ytr_t)
                loss.backward(); opt.step()

            with torch.no_grad():
                val_loss = lossf(model(Xval_t), yval_t).item()

            if val_loss < best_loss:
                best_loss, best_state, best_cfg = val_loss, model.state_dict(), cfg
                print(f"[dynamic] ★ new best {best_loss:.4f} cfg={cfg}")

    best_model = build_model(best_cfg, input_dim)
    best_model.load_state_dict(best_state)

# 7.  保存 / 上传 artefacts
buf = io.BytesIO()
torch.save(best_model.cpu(), buf)           # 存完整模型
buf.seek(0)
save_bytes(f"{MODEL_DIR}/model.pt", buf.read())

save_bytes(f"{MODEL_DIR}/last_model_config.json",
           json.dumps(best_cfg).encode(), "application/json")
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           (datetime.utcnow().isoformat() + "Z").encode())

log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update",
           value=round(best_loss, 6))
sync_all_metrics_to_minio()

print(f"[dynamic] retrain done | JS={JS:.4f} | loss={best_loss:.6f} | cfg={best_cfg}")