#!/usr/bin/env python3
"""
ml.dynamic_retrain ─ JS 触发的在线增量重训练  
新的实现：——最近 N 条 + 少量 baseline 混合采样  
——Early Stopping 校验  
——生成 metrics_tmp.json & model_tmp.pt 并更新 latest.txt
"""
import os
import sys
import io
import json
import itertools
import joblib
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, JS_TRIGGER_THRESH, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.features     import FEATURE_COLS
from shared.utils        import calculate_accuracy_within_threshold
from ml.model            import build_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler     import Timer

# 1. 当前漂移值（由 monitor.py 传参）
JS = float(sys.argv[1])

# ── 2. 合并累计样本 ────────────────────────────────────────
latest_rows = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True).tolist()
cumu_path   = "/mnt/pvc/all_seen.npy"
if os.path.exists(cumu_path):
    all_seen = np.load(cumu_path, allow_pickle=True).tolist()
else:
    all_seen = []

def _row_key(r):
    feats = tuple(r["features"][c] for c in FEATURE_COLS)
    return feats + (r["label"],)

# 去重累积
merged = { _row_key(r): r for r in all_seen }
merged.update({ _row_key(r): r for r in latest_rows })
all_seen = list(merged.values())
np.save(cumu_path, np.array(all_seen, dtype=object))

# ── 3. 采样：最近 N + 少量 baseline ─────────────────────────
RECENT_N = int(os.getenv("RETRAIN_RECENT_N", "1500"))
recent_rows = latest_rows[-RECENT_N:]

# 混入 10% baseline 保证不过拟合新分布
baseline_sample = []
if all_seen:
    # 希望抽取的目标数量
    k_target = max(1, int(0.1 * RECENT_N))
    # 关键修复：实际抽取数不能超过已有样本数
    k = min(k_target, len(all_seen))
    baseline_sample = list(np.random.choice(all_seen, size=k, replace=False))

# 合并成最终训练集
all_rows = recent_rows + baseline_sample

# ── 4. 加载 scaler & PCA（保证 6-维输入） ─────────────────────
buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
)
scaler = joblib.load(buf)

buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/pca.pkl")["Body"].read()
)
pca = joblib.load(buf)

# ── 5. 构造训练 / 验证集 ─────────────────────────────────────
X_all = np.asarray(
    [[r["features"][c] for c in FEATURE_COLS] for r in all_rows],
    dtype=np.float32
)
y_all = np.asarray([r["label"] for r in all_rows], dtype=np.float32)

X_scaled = scaler.transform(X_all)
X_pca    = pca.transform(X_scaled).astype(np.float32)
input_dim = X_pca.shape[1]

Xtr, Xval, ytr, yval = train_test_split(
    X_pca, y_all, test_size=0.2, random_state=0
)
Xtr_t, ytr_t   = torch.from_numpy(Xtr),  torch.from_numpy(ytr)
Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)

# ── 6. 四档策略：跳过 / 快速分支 / 网格搜索 ─────────────────────
param_grid_A = {  # 轻度漂移
    "learning_rate": [1e-3],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32)],
    "activation":    ["relu"],
}
param_grid_B = {  # 中度漂移
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [8, 16, 32],
    "hidden_layers": [(64, 32), (128, 64, 32)],
    "activation":    ["relu"],
}
param_grid_C = {  # 严重漂移
    "learning_rate": [1e-2, 1e-3],
    "batch_size":    [8, 16],
    "hidden_layers": [(128, 64, 32)],
    "activation":    ["relu", "gelu"],
    "loss":          ["Huber", "mse"],
}

VERY_LIGHT_THR = JS_TRIGGER_THRESH
LIGHT_THR      = JS_SEV1_THRESH
MID_THR        = JS_SEV2_THRESH

def _grid_by_js(js_val: float):
    if js_val <= VERY_LIGHT_THR:
        return None
    if js_val <= LIGHT_THR:
        return param_grid_A
    if js_val <= MID_THR:
        return param_grid_B
    return param_grid_C

grid_cfg = _grid_by_js(JS)
if grid_cfg is None:
    print(f"[dynamic] JS={JS:.4f} 低于 {VERY_LIGHT_THR:.3f}，跳过重训")
    sys.exit(0)

device = "cpu"
LOSS_MAP = {"Huber": nn.SmoothL1Loss, "mse": nn.MSELoss}

# 6.1 快速分支：带 Early Stopping
if grid_cfg is param_grid_A:
    best_cfg = {
        "hidden_layers": (64, 32),
        "activation":    "relu",
        "learning_rate": 1e-3,
    }
    model = build_model(best_cfg, input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=best_cfg["learning_rate"])
    lossf = nn.SmoothL1Loss()

    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(1, 51):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            val_loss = lossf(model(Xval_t), yval_t).item()

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, patience = val_loss, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 8:
                break

    model.load_state_dict(best_state)
    best_model = model.eval()

# 6.2 网格搜索分支 (B / C)
else:
    best_loss, best_state, best_cfg = float("inf"), None, None
    with Timer("Dynamic_Retrain", "retrain"):
        for hp in itertools.product(*grid_cfg.values()):
            cfg = dict(zip(grid_cfg.keys(), hp))
            model = build_model(cfg, input_dim).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            lossf = LOSS_MAP.get(cfg.get("loss","Huber"), nn.SmoothL1Loss)()

            for _ in range(10):
                opt.zero_grad()
                loss = lossf(model(Xtr_t), ytr_t)
                loss.backward()
                opt.step()

            with torch.no_grad():
                val_loss = lossf(model(Xval_t), yval_t).item()

            if val_loss < best_loss:
                best_loss, best_state, best_cfg = val_loss, model.state_dict(), cfg
                print(f"[dynamic] new best {best_loss:.4f} cfg={best_cfg}")

    best_model = build_model(best_cfg, input_dim)
    best_model.load_state_dict(best_state)

# ── 7. 保存／上传 artefacts ────────────────────────────────────
# 7.1 评估指标写入 JSON
with torch.no_grad():
    y_hat = best_model(Xval_t).cpu().numpy()
acc15 = calculate_accuracy_within_threshold(yval, y_hat, 0.15)

metrics = {
    "js_trigger":    JS,
    "val_loss":      float(best_loss),
    "acc@0.15":      float(acc15),
    "trained_on":    len(all_rows),
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
}
save_bytes(f"{MODEL_DIR}/metrics_tmp.json",
           json.dumps(metrics).encode(),
           "application/json")

# 7.2 模型保存到临时文件
buf = io.BytesIO()
torch.save(best_model.cpu(), buf)
buf.seek(0)
save_bytes(f"{MODEL_DIR}/model_tmp.pt",
           buf.read(),
           "application/octet-stream")

# 7.3 原子更新 latest.txt
latest_content = "model_tmp.pt\nmetrics_tmp.json"
save_bytes(f"{MODEL_DIR}/latest.txt",
           latest_content.encode(),
           "text/plain")

# 7.4 上报指标 & 同步到 MinIO
log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update", value=round(best_loss,6))
sync_all_metrics_to_minio()

print(f"[dynamic] retrain done | JS={JS:.4f} | loss={best_loss:.6f} | cfg={best_cfg}")
