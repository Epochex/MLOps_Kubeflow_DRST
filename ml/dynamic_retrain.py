#!/usr/bin/env python3
"""
ml.dynamic_retrain ─ JS 触发的在线增量重训练  
新的实现：——最近 N 条 + 少量 baseline 混合采样  
——Early Stopping 校验  
——生成 metrics_tmp.json & model_tmp.pt 并更新 latest.txt
只使用与 output_rate 相关性最高的 10 维特征 + StandardScaler，不做 PCA。
"""
import os
import sys
import io
import json
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split

from shared.minio_helper import s3, BUCKET, save_bytes, load_csv
from shared.config       import MODEL_DIR, JS_TRIGGER_THRESH, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.utils        import calculate_accuracy_within_threshold
from ml.model            import build_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler     import Timer
from sklearn.preprocessing import StandardScaler

from shared.features     import FEATURE_COLS
from shared.config       import TARGET_COL


# ------------------------ 参数 & 时间戳 ------------------------
start_ts = time.time()
# 来自 monitor.py 以命令行参数传入的漂移值
JS = float(sys.argv[1])

# 只保留最新 N 条样本用于训练
RECENT_N = int(os.getenv("RETRAIN_RECENT_N", "300"))

# Severity→early-stopping 容忍度
PATIENCE_K = {1: 2, 2: 4, 3: 6}

# ------------------------ 加载最新窗口 ------------------------
latest_path = "/tmp/monitor/latest_batch.npy"
if not os.path.exists(latest_path):
    raise FileNotFoundError(f"[dynamic] missing {latest_path}")

rows = np.load(latest_path, allow_pickle=True).tolist()
if not rows:
    raise ValueError("[dynamic] latest_batch.npy is empty")

# 截尾到 RECENT_N 条
if len(rows) > RECENT_N:
    rows = rows[-RECENT_N:]
print(f"[dynamic] loaded {len(rows)} samples")

# ------------------------ 构造训练数据 ------------------------
# 1) Top-10 特征列表
raw_feats = s3.get_object(Bucket=BUCKET,
                          Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
SELECTED_FEATS = json.loads(raw_feats)

# 2) 从 Offline 拿来的原始 scaler（只用于 baseline 对比）
buf_old = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
)
scaler_old = joblib.load(buf_old)

# 3) 准备新数据矩阵
X_all = np.vstack([
    [r["features"].get(c, 0.0) for c in SELECTED_FEATS]
    for r in rows
]).astype(np.float32)
y_all = np.asarray([r["label"] for r in rows], dtype=np.float32)

# 4) 重新 fit 新 scaler，并上传
scaler_new = StandardScaler().fit(X_all)
buf_new = io.BytesIO()
joblib.dump(scaler_new, buf_new)
buf_new.seek(0)
save_bytes(f"{MODEL_DIR}/scaler_tmp.pkl", buf_new.read(), "application/octet-stream")

X_scaled = scaler_new.transform(X_all).astype(np.float32)

# 5) 划分训练/验证
Xtr, Xval, ytr, yval = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)
Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)

# ------------------------ 根据 JS 定 Severity & 超参 ------------------------
if   JS <= JS_SEV1_THRESH: sev = 1
elif JS <= JS_SEV2_THRESH: sev = 2
else:                      sev = 3

cfg_map = {
    1: dict(hidden_layers=(64,32),    activation="relu", learning_rate=5e-4, batch_size=16, loss="smooth_l1"),
    2: dict(hidden_layers=(128,64,32),activation="relu", learning_rate=1e-3, batch_size=16, loss="smooth_l1"),
    3: dict(hidden_layers=(256,128,64),activation="gelu", learning_rate=1e-3, batch_size=16, loss="mse"),
}
MAX_EPOCH = {1:15, 2:20, 3:30}[sev]
PATIENCE  = PATIENCE_K[sev]
cfg       = cfg_map[sev]

print(f"[dynamic] JS={JS:.4f} → severity={sev}, cfg={cfg}, max_epoch={MAX_EPOCH}, patience={PATIENCE}")

# ------------------------ 定义训练函数 ------------------------
def _train_once(cfg: dict, freeze_last: bool):
    # Warm-start 从 baseline_model.pt
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
    model = torch.load(io.BytesIO(raw), map_location="cpu").train()

    # 冻结策略
    if freeze_last:
        for p in model.parameters(): p.requires_grad = False
        for p in model.net[-1].parameters(): p.requires_grad = True
    else:
        for p in model.parameters(): p.requires_grad = True

    # 损失 & 优化器
    loss_fn = nn.MSELoss() if cfg["loss"]=="mse" else nn.SmoothL1Loss()
    optim   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"]
    )

    best_state, best_loss, es = None, float("inf"), 0
    for epoch in range(1, MAX_EPOCH+1):
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i+cfg["batch_size"]]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()

        with torch.no_grad():
            vl = loss_fn(model(Xval_t), yval_t).item()
        if vl + 1e-6 < best_loss:
            best_loss, best_state, es = vl, model.state_dict(), 0
        else:
            es += 1
            if es >= PATIENCE:
                print(f"[dynamic] early stop at epoch={epoch}, val_loss={vl:.6f}")
                break

    model.load_state_dict(best_state)
    return model.eval(), best_loss

# ------------------------ 执行训练 ------------------------
freeze_last = (sev == 1)
best_model, best_loss = _train_once(cfg, freeze_last)
print(f"[dynamic] retrained loss={best_loss:.6f}")

# ------------------------ 评估 & 上报 ------------------------
# baseline vs new on same 验证集
raw_base = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
base_mdl = torch.load(io.BytesIO(raw_base), map_location="cpu").eval()
with torch.no_grad():
    base_acc = calculate_accuracy_within_threshold(yval, base_mdl(Xval_t).cpu().numpy(), 0.15)
    new_acc  = calculate_accuracy_within_threshold(yval, best_model(Xval_t).cpu().numpy(), 0.15)

metrics = {
    "js_trigger":         JS,
    "val_loss":           float(best_loss),
    "acc@0.15":           new_acc,
    "baseline_acc@0.15":  base_acc,
    "trained_on":         len(rows),
    "timestamp_utc":      datetime.utcnow().isoformat()+"Z",
}
save_bytes(f"{MODEL_DIR}/metrics_tmp.json", json.dumps(metrics).encode(), "application/json")

buf_mod = io.BytesIO()
torch.save(best_model.cpu(), buf_mod); buf_mod.seek(0)
save_bytes(f"{MODEL_DIR}/model_tmp.pt", buf_mod.read(), "application/octet-stream")

# 原子更新 latest.txt，驱动 hot-reload
latest_txt = "model_tmp.pt\nmetrics_tmp.json\nscaler_tmp.pkl"
save_bytes(f"{MODEL_DIR}/latest.txt", latest_txt.encode(), "text/plain")

log_metric(component="retrain", event="model_pushed")
sync_all_metrics_to_minio()


elapsed = time.time() - start_ts
print(
    f"[dynamic] retrain done | JS={JS:.4f} | "
    f"loss={best_loss:.6f} | acc@0.15(old→new)={baseline_acc:.2f}%→{new_acc:.2f}% | "
    f"cfg={best_cfg} | elapsed={elapsed:.2f}s"
)