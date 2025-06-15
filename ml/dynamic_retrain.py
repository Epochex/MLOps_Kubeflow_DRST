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

import time
start_ts = time.time()

# 1. 当前漂移值（由 monitor.py 传参）
JS = float(sys.argv[1])

# ── 2. 从本地临时盘读取最新窗口 & 定位累积文件 ────────────────────────────
import os

# Monitor 写入的本地文件
latest_path = "/tmp/monitor/latest_batch.npy"
if not os.path.exists(latest_path):
    raise FileNotFoundError(f"latest batch not found: {latest_path}")
latest_rows = np.load(latest_path, allow_pickle=True).tolist()

# 本地累积也放到同一路径
cumu_path = "/tmp/monitor/all_seen.npy"
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

# ───────────────────────── 4. 预处理：60 维特征 ─────────────────────────
buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
)
scaler = joblib.load(buf)

# —— 新增：如果最新窗口里真正“新的”样本不足，就暂缓重训 ——
NEW_SAMPLE_MIN = int(os.getenv("NEW_SAMPLE_MIN", "500"))   # 可在 k8s 环境变量里改
if len(latest_rows) < NEW_SAMPLE_MIN:
    print(f"[dynamic] only {len(latest_rows)} recent rows (<{NEW_SAMPLE_MIN}), postpone retrain")
    sys.exit(0)

# —— 把 all_rows 映射成 60 维 Scaler 特征 ——
X_all = np.asarray(
    [[r["features"][c] for c in FEATURE_COLS] for r in all_rows],
    dtype=np.float32
)
y_all = np.asarray([r["label"] for r in all_rows], dtype=np.float32)

X_scaled = scaler.transform(X_all).astype(np.float32)
input_dim = X_scaled.shape[1]          # = 60

# —— 划分训练 / 验证集 ——
Xtr, Xval, ytr, yval = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)
Xtr_t, ytr_t   = torch.from_numpy(Xtr),  torch.from_numpy(ytr)
Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)

# ───────────────────────── 6. 轻量多档训练 ─────────────────────────
"""
思路  
▪ 仍然分 A / B / C 三档，但每档只保留极少量候选超参，  
  以保证 1500 条样本时训练 < 25 s。  
▪ 每个候选配置用统一的 _train_one() 循环，内置 Early-Stopping( patience=6 )。  
"""

# ---------- 6.0 三档轻量网格 ----------
param_grid_A = [  # JS ∈ (TRIGGER, SEV1]
    {"hidden_layers": (128, 64),      "activation": "relu",
     "learning_rate": 5e-4,           "batch_size": 32}
]

param_grid_B = [  # SEV1 < JS ≤ SEV2
    {"hidden_layers": (128, 64, 32),  "activation": "relu",
     "learning_rate": 5e-4,           "batch_size": 32},
    {"hidden_layers": (128, 64),      "activation": "relu",
     "learning_rate": 1e-3,           "batch_size": 32},
]

param_grid_C = [  # JS > SEV2
    {"hidden_layers": (256, 128, 64), "activation": "relu",
     "learning_rate": 1e-3,           "batch_size": 32},
    {"hidden_layers": (256, 128, 64), "activation": "gelu",
     "learning_rate": 1e-3,           "batch_size": 32},
]

# ---------- 6.1 根据 JS 选搜索空间 ----------
if JS <= JS_SEV1_THRESH:
    search_space = param_grid_A
elif JS <= JS_SEV2_THRESH:
    search_space = param_grid_B
else:
    search_space = param_grid_C

device      = "cpu"                          # retrain 容器一般无 GPU
loss_fn     = nn.SmoothL1Loss()              # 统一使用 Huber
PATIENCE    = 6                              # Early-Stopping
MAX_EPOCH   = 50

def _train_one(cfg: dict) -> tuple[nn.Module, float]:
    """单配置训练并返回 (model, best_val_loss)"""
    model = build_model(cfg, input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_state, best_loss, es = None, float("inf"), 0
    for epoch in range(1, MAX_EPOCH + 1):
        # --- Mini-batch 训练 ---
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), cfg["batch_size"]):
            xb = torch.from_numpy(Xtr[perm[i:i+cfg["batch_size"]]]).to(device)
            yb = torch.from_numpy(ytr[perm[i:i+cfg["batch_size"]]]).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # --- 验证 ---
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.from_numpy(Xval).to(device)),
                torch.from_numpy(yval).to(device)
            ).item()

        if val_loss < best_loss - 1e-4:           # 有明显提升
            best_loss, best_state, es = val_loss, model.state_dict(), 0
        else:                                     # 无提升
            es += 1
            if es >= PATIENCE:
                break

    model.load_state_dict(best_state)             # 回到最佳点
    return model.eval(), best_loss


# ---------- 6.2 遍历搜索空间，选最佳 ----------
best_model, best_loss, best_cfg = None, float("inf"), None
for cfg in search_space:
    mdl, vloss = _train_one(cfg)
    if vloss < best_loss:
        best_model, best_loss, best_cfg = mdl, vloss, cfg
        print(f"[dynamic] new best {vloss:.4f}  cfg={best_cfg}")

# 若因某种原因没有训练出模型（极小概率），直接退出
if best_model is None:
    print("[dynamic] no valid model trained, exit")
    sys.exit(0)


# ── 7. 保存／上传 artefacts ────────────────────────────────────
# 7.1 评估指标写入 JSON

# ---------- baseline_model 在同一验证集上的 acc ------------------------
base_raw = s3.get_object(Bucket=BUCKET,
                         Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
baseline_model = torch.load(io.BytesIO(base_raw), map_location="cpu").eval()
with torch.no_grad():
    base_hat = baseline_model(Xval_t).cpu().numpy()
baseline_acc = calculate_accuracy_within_threshold(yval, base_hat, 0.15)

metrics = {
    "js_trigger":         JS,
    "val_loss":           float(best_loss),
    "acc@0.15":           float(acc15),        # 新模型
    "baseline_acc@0.15":  float(baseline_acc), # baseline 模型
    "trained_on":         len(all_rows),
    "timestamp_utc":      datetime.utcnow().isoformat() + "Z",
}
save_bytes(f"{MODEL_DIR}/metrics_tmp.json",
           json.dumps(metrics).encode(),
           "application/json")

# --- 在保存前，把验证 acc 写进模型本体 ---
best_model._val_acc15 = float(acc15)

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

elapsed = time.time() - start_ts
print(
    f"[dynamic] retrain done | JS={JS:.4f} | "
    f"loss={best_loss:.6f} | cfg={best_cfg} | "
    f"elapsed={elapsed:.2f}s"
)