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

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, JS_TRIGGER_THRESH, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.utils        import calculate_accuracy_within_threshold
from ml.model            import build_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler     import Timer


# ---------------------------------------------------------------------
# 0. 超参数门槛
# ---------------------------------------------------------------------
NEW_SAMPLE_MIN = int(os.getenv("NEW_SAMPLE_MIN", "200"))   # ↓ 300 → 200
RECENT_N       = int(os.getenv("RETRAIN_RECENT_N", "1500"))

start_ts = time.time()

# 1. 当前漂移值（由 monitor.py 传参）
JS = float(sys.argv[1])

# ---------------------------------------------------------------------
# 2. 读取最新窗口 & 累积去重
# ---------------------------------------------------------------------
latest_path = "/tmp/monitor/latest_batch.npy"
if not os.path.exists(latest_path):
    raise FileNotFoundError(f"latest batch not found: {latest_path}")
latest_rows = np.load(latest_path, allow_pickle=True).tolist()

cumu_path = "/tmp/monitor/all_seen.npy"
if os.path.exists(cumu_path):
    all_seen = np.load(cumu_path, allow_pickle=True).tolist()
else:
    all_seen = []

def _row_key(r):
    # 用所有特征 + label 去重
    feats = tuple(r["features"][c] for c in r["features"].keys())
    return feats + (r["label"],)

merged = { _row_key(r): r for r in all_seen }
merged.update({ _row_key(r): r for r in latest_rows })
all_seen = list(merged.values())
np.save(cumu_path, np.array(all_seen, dtype=object))

# ---------------------------------------------------------------------
# 3. 采样：
# ---------------------------------------------------------------------
# RECENT_N = int(os.getenv("RETRAIN_RECENT_N", "1500"))
# recent_rows = latest_rows[-RECENT_N:]

# baseline_sample = []
# if all_seen:
#     k_target = max(1, int(0.1 * RECENT_N))
#     k = min(k_target, len(all_seen))
#     baseline_sample = list(np.random.choice(all_seen, size=k, replace=False))

all_rows = recent_rows + baseline_sample

# ---------------------------------------------------------------------
# 4. 加载 Top-10 特征列表 & StandardScaler
# ---------------------------------------------------------------------
# 4.1 Top-10 特征
raw_feats = s3.get_object(Bucket=BUCKET,
                          Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
SELECTED_FEATS = json.loads(raw_feats)

# 4.2 StandardScaler（offline 已训练好）
buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()
)
scaler = joblib.load(buf)

# ---------------------------------------------------------------------
# 5. 数据充足性检查
# ---------------------------------------------------------------------
if len(latest_rows) < NEW_SAMPLE_MIN:
    print(f"[dynamic] only {len(latest_rows)} recent rows (<{NEW_SAMPLE_MIN}), postpone retrain")
    sys.exit(0)


# 特征矩阵
X_all = np.asarray(
    [[r["features"].get(c, 0.0) for c in SELECTED_FEATS]
     for r in all_rows],
    dtype=np.float32
)
y_all = np.asarray([r["label"] for r in all_rows], dtype=np.float32)

# 归一化
X_scaled = scaler.transform(X_all).astype(np.float32)
input_dim = X_scaled.shape[1]  # =10

# 划分训练/验证
Xtr, Xval, ytr, yval = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)
Xtr_t, ytr_t   = torch.from_numpy(Xtr),  torch.from_numpy(ytr)
Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)

# ---------------------------------------------------------------------
# 6. 动态网格微调 + Early Stopping (warm-start baseline)
# ---------------------------------------------------------------------
# 6.1 定义搜索空间（保持原有三档）
param_grid_A = [
    {"hidden_layers": (32,),          "activation": "relu",
     "learning_rate": 1e-3,           "batch_size": 64, "loss": "smooth_l1"},
]
param_grid_B = [
    {"hidden_layers": (64, 32),       "activation": "relu",
     "learning_rate": 5e-3,           "batch_size": 64, "loss": "smooth_l1"},
]
param_grid_C = [
    {"hidden_layers": (128, 64, 32),  "activation": "relu",
     "learning_rate": 1e-2,           "batch_size": 32, "loss": "smooth_l1"},
    {"hidden_layers": (256, 128, 64), "activation": "gelu",
     "learning_rate": 5e-3,           "batch_size": 32, "loss": "mse"},
]

# 6.2 根据漂移严重度选择网格 & 微调节奏
if JS <= JS_SEV1_THRESH:
    search_space = param_grid_A
    MAX_EPOCH = 5     # A 档：超轻量
    PATIENCE  = 2
elif JS <= JS_SEV2_THRESH:
    search_space = param_grid_B
    MAX_EPOCH = 10    # B 档：中等
    PATIENCE  = 3
else:
    search_space = param_grid_C
    MAX_EPOCH = 50    # C 档：全量强训
    PATIENCE  = 6

device = "cpu"

def _train_one(cfg: dict) -> tuple[torch.nn.Module, float]:
    # — ① 从 baseline warm-start —
    raw = s3.get_object(Bucket=BUCKET,
                        Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
    model = torch.load(io.BytesIO(raw), map_location=device).train()

    # — ② 损失 & 优化器 —
    loss_fn = (nn.MSELoss() if cfg.get("loss") == "mse"
               else nn.SmoothL1Loss())
    opt     = torch.optim.Adam(model.parameters(),
                               lr=cfg["learning_rate"])

    best_state, best_loss, es = None, float("inf"), 0
    # — ③ 按 cfg.batch_size 训练 —
    for epoch in range(1, MAX_EPOCH + 1):
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            xb  = torch.from_numpy(Xtr[idx]).to(device)
            yb  = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()

        # — ④ 验证集 Early Stopping —
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.from_numpy(Xval).to(device)),
                torch.from_numpy(yval).to(device)
            ).item()

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, es = val_loss, model.state_dict(), 0
        else:
            es += 1
            if es >= PATIENCE:
                break

    # — ⑤ 恢复最优权重 & 切 eval —
    model.load_state_dict(best_state)
    return model.eval(), best_loss

# 6.3 执行网格搜索
best_model, best_loss, best_cfg = None, float("inf"), None
for cfg in search_space:
    mdl, loss_val = _train_one(cfg)
    if loss_val < best_loss:
        best_model, best_loss, best_cfg = mdl, loss_val, cfg
        print(f"[dynamic] new best {loss_val:.4f}  cfg={best_cfg}")

if best_model is None:
    print("[dynamic] no valid model trained, exit")
    sys.exit(0)

# ---------------------------------------------------------------------
# 7. 保存 artefacts & 更新 latest.txt
# ---------------------------------------------------------------------
# 7.1 评估 baseline_model 在相同验证集上的表现
base_raw = s3.get_object(Bucket=BUCKET,
                         Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
baseline_model = torch.load(io.BytesIO(base_raw), map_location="cpu").eval()

with torch.no_grad():
    base_hat = baseline_model(Xval_t).detach().cpu().numpy()
baseline_acc = calculate_accuracy_within_threshold(yval, base_hat, 0.15)

# 7.2 评估新训练模型在验证集上的表现
with torch.no_grad():
    new_hat = best_model(Xval_t).detach().cpu().numpy()
new_acc = calculate_accuracy_within_threshold(yval, new_hat, 0.15)

metrics = {
    "js_trigger":         JS,
    "val_loss":           float(best_loss),
    "acc@0.15":           float(new_acc),
    "baseline_acc@0.15":  float(baseline_acc),
    "trained_on":         len(all_rows),
    "timestamp_utc":      datetime.utcnow().isoformat() + "Z",
}
save_bytes(f"{MODEL_DIR}/metrics_tmp.json",
           json.dumps(metrics).encode(),
           "application/json")

# 在模型对象中记录新模型的验证准确率
best_model._val_acc15 = new_acc

# 7.3 保存新模型
buf = io.BytesIO()
torch.save(best_model.cpu(), buf)
buf.seek(0)
save_bytes(f"{MODEL_DIR}/model_tmp.pt",
           buf.read(),
           "application/octet-stream")

# 7.4 原子更新 latest.txt，让 inference_consumer 热加载
latest_content = "model_tmp.pt\nmetrics_tmp.json"
save_bytes(f"{MODEL_DIR}/latest.txt",
           latest_content.encode(),
           "text/plain")

# 7.5 上报 & 同步所有指标
log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update", value=round(best_loss, 6))
sync_all_metrics_to_minio()

elapsed = time.time() - start_ts
print(
    f"[dynamic] retrain done | JS={JS:.4f} | "
    f"loss={best_loss:.6f} | acc@0.15(old→new)={baseline_acc:.2f}%→{new_acc:.2f}% | "
    f"cfg={best_cfg} | elapsed={elapsed:.2f}s"
)