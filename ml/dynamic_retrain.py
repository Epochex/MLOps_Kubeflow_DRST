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
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
# 0. 依赖与环境变量
# ---------------------------------------------------------------------

NEW_SAMPLE_MIN   = int(os.getenv("NEW_SAMPLE_MIN",  "100"))   # 快速重训阈值
RETRAIN_RECENT_N = int(os.getenv("RETRAIN_RECENT_N", "300"))  # 完整重训窗口
JS               = float(sys.argv[1])                         # monitor 传入
TMP_DIR          = "/tmp/monitor"
LATEST_NPY       = f"{TMP_DIR}/latest_batch.npy"
CUMU_NPY         = f"{TMP_DIR}/all_seen.npy"
start_ts         = time.time()


start_ts = time.time()

# ---------------------------------------------------------------------
# 1. 读取与去重最近窗口
# ---------------------------------------------------------------------
if not os.path.exists(LATEST_NPY):
    raise FileNotFoundError(f"latest batch not found: {LATEST_NPY}")

latest_rows: List[dict] = np.load(LATEST_NPY, allow_pickle=True).tolist()

# 累积去重
if os.path.exists(CUMU_NPY):
    all_seen: List[dict] = np.load(CUMU_NPY, allow_pickle=True).tolist()
else:
    all_seen = []

merge_key = lambda r: (tuple(sorted(r["features"].items())), r["label"])
merged: Dict[Tuple, dict] = {merge_key(r): r for r in all_seen}
for r in latest_rows:
    merged[merge_key(r)] = r
all_seen = list(merged.values())
np.save(CUMU_NPY, np.array(all_seen, dtype=object))


# ---------------------------------------------------------------------
# 2. 判定训练模式（quick / full）
#    monitor 已保证 latest_rows ≥ NEW_SAMPLE_MIN；此处不再等待
# ---------------------------------------------------------------------
if len(latest_rows) >= RETRAIN_RECENT_N:
    mode       = "full"
    train_rows = latest_rows[-RETRAIN_RECENT_N:]
else:
    mode       = "quick"
    train_rows = latest_rows[-NEW_SAMPLE_MIN:]

print(f"[dynamic] mode={mode}  rows={len(train_rows)}  JS={JS:.4f}")


# ---------------------------------------------------------------------
# 4. 准备数据：特征列表 & StandardScaler
# ---------------------------------------------------------------------
raw_feats = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/selected_feats.json")["Body"].read()
SELECTED_FEATS = json.loads(raw_feats)

scaler_buf = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(scaler_buf)

X = np.array([[r["features"].get(c, 0.0) for c in SELECTED_FEATS] for r in train_rows],
             dtype=np.float32)
y = np.array([r["label"] for r in train_rows], dtype=np.float32)
X_scaled = scaler.transform(X)

Xtr, Xval, ytr, yval = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
Xtr_t, Xval_t = map(torch.from_numpy, (Xtr, Xval))
ytr_t, yval_t = map(torch.from_numpy, (ytr, yval))

device = "cpu"

# # ---------------------------------------------------------------------
# # 5. 定义网格
# # ---------------------------------------------------------------------
# grid_A = [
#     {"hidden_layers": (32,),        "lr": 1e-3, "bs": 64, "act": "relu", "loss": "smooth_l1", "wd": 1e-4},
#     {"hidden_layers": (16,),        "lr": 5e-4, "bs": 64, "act": "relu", "loss": "smooth_l1", "wd": 1e-4},
# ]
# grid_B = [
#     {"hidden_layers": (64, 32),     "lr": 5e-4, "bs": 64, "act": "relu", "loss": "smooth_l1", "wd": 5e-4},
#     {"hidden_layers": (64, 32),     "lr": 1e-3, "bs": 32, "act": "gelu", "loss": "smooth_l1", "wd": 5e-4},
# ]
# grid_C = [
#     {"hidden_layers": (128,64,32),  "lr": 1e-3, "bs": 32, "act": "relu", "loss": "smooth_l1", "wd": 1e-3},
#     {"hidden_layers": (256,128,64), "lr": 5e-4, "bs": 16, "act": "gelu", "loss": "mse",      "wd": 1e-3},
# ]

# if mode == "quick":
#     search_space, MAX_EPOCH, PATIENCE = grid_A, 15, 4
# elif JS <= JS_SEV1_THRESH:
#     search_space, MAX_EPOCH, PATIENCE = grid_A, 15, 4
# elif JS <= JS_SEV2_THRESH:
#     search_space, MAX_EPOCH, PATIENCE = grid_B, 30, 6
# else:
#     search_space, MAX_EPOCH, PATIENCE = grid_C, 60, 8
# ---------------------------------------------------------------------
# 5. 网格配置（完全对应表格中的 Severity-1/2/K 组合）
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 5. 网格配置（精简版，避免长时间阻塞 monitor）
# ---------------------------------------------------------------------

# Quick（≤300 条）：单层 MLP，保证 ≤10 epoch 就能出结果
grid_A = [
    {"hidden_layers": (32,), "act": "relu", "lr": 1e-3,
     "bs": 64, "loss": "smooth_l1", "wd": 1e-4},
]

# Severity-1：6 组 → 2 组（单一结构 × 2 个学习率）
grid_S1 = [
    {"hidden_layers": (64, 32), "act": "relu", "lr": lr,
     "bs": 32, "loss": "smooth_l1", "wd": 5e-4}
    for lr in (1e-3, 5e-4)
]

# Severity-2：16 组 → 3 组（两种隐藏层 + 一个较稳健的 LR）
grid_S2 = [
    {"hidden_layers": h, "act": "relu", "lr": 5e-4,
     "bs": 32, "loss": "smooth_l1", "wd": 5e-4}
    for h in ((64, 32), (128, 64, 32))
]

# Severity-K：48 组 → 4 组（两种深网 × 两种激活）
grid_SK = [
    {"hidden_layers": h, "act": a, "lr": 1e-3,
     "bs": 32, "loss": "smooth_l1", "wd": 1e-3}
    for h in ((256, 128, 64), (128, 128, 64, 32))
    for a in ("relu", "gelu")
]

# -------- 新的训练轮数/耐心设置 --------
if mode == "quick":
    search_space, MAX_EPOCH, PATIENCE = grid_A, 10, 3
elif JS <= JS_SEV1_THRESH:
    search_space, MAX_EPOCH, PATIENCE = grid_S1, 15, 4
elif JS <= JS_SEV2_THRESH:
    search_space, MAX_EPOCH, PATIENCE = grid_S2, 20, 5
else:
    search_space, MAX_EPOCH, PATIENCE = grid_SK, 30, 6

# ---------------------------------------------------------------------
# 6. 训练函数
# ---------------------------------------------------------------------
def _loss_fn(name: str):
    return nn.MSELoss() if name == "mse" else nn.SmoothL1Loss()

def _warm_start_model(cfg: dict) -> torch.nn.Module:
    """
    重训阶段直接根据当前超参 **动态构造** 一个新网络，
    不再从 baseline 权重 warm-start，以避免结构不匹配。
    """
    model = build_model(
        {"hidden_layers": cfg["hidden_layers"], "activation": cfg["act"]},
        input_dim=Xtr.shape[1]
    ).to(device).train()
    return model

def _train(cfg: dict) -> Tuple[torch.nn.Module, float]:
    mdl = _warm_start_model(cfg)
    opt = torch.optim.Adam(mdl.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = _loss_fn(cfg["loss"])
    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCH + 1):
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), cfg["bs"]):
            idx = perm[i:i+cfg["bs"]]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad(); loss = lossf(mdl(xb), yb); loss.backward(); opt.step()

        with torch.no_grad():
            val = lossf(mdl(Xval_t.to(device)), yval_t.to(device)).item()

        if val < best_loss - 1e-4:
            best_loss, best_state, no_imp = val, mdl.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    mdl.load_state_dict(best_state)
    return mdl.eval(), best_loss

# ---------------------------------------------------------------------
# 7. 网格搜索
# ---------------------------------------------------------------------
best_mdl, best_loss, best_cfg = None, float("inf"), None
for cfg in search_space:
    mdl, loss_val = _train(cfg)
    if loss_val < best_loss:
        best_mdl, best_loss, best_cfg = mdl, loss_val, cfg
        print(f"[dynamic] new best {loss_val:.4f}  cfg={best_cfg}")

if best_mdl is None:
    print("[dynamic] no valid model — exit")
    sys.exit(0)

# ---------------------------------------------------------------------
# 8. 评估准确率
# ---------------------------------------------------------------------
base_raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/baseline_model.pt")["Body"].read()
baseline_mdl = torch.load(io.BytesIO(base_raw), map_location="cpu").eval()

with torch.no_grad():
    new_hat  = best_mdl(Xval_t).numpy()
    base_hat = baseline_mdl(Xval_t).numpy()

new_acc  = calculate_accuracy_within_threshold(yval, new_hat , 0.15)
base_acc = calculate_accuracy_within_threshold(yval, base_hat, 0.15)

# ---------------------------------------------------------------------
# 9. artefacts 上传 & latest.txt 更新
# ---------------------------------------------------------------------
best_mdl._val_acc15 = float(new_acc)

buf = io.BytesIO(); torch.save(best_mdl.cpu(), buf); buf.seek(0)
save_bytes(f"{MODEL_DIR}/model_tmp.pt", buf.read(), "application/octet-stream")

metrics = {
    "js_trigger":        JS,
    "mode":              mode,
    "val_loss":          float(best_loss),
    "acc@0.15":          float(new_acc),
    "baseline_acc@0.15": float(base_acc),
    "trained_on":         len(train_rows),
    "timestamp_utc":     datetime.utcnow().isoformat() + "Z",
}
save_bytes(f"{MODEL_DIR}/metrics_tmp.json",
           json.dumps(metrics).encode(),
           "application/json")

save_bytes(f"{MODEL_DIR}/latest.txt",
           b"model_tmp.pt\nmetrics_tmp.json",
           "text/plain")

log_metric(component="retrain", event="model_update", val_loss=round(best_loss,6))
sync_all_metrics_to_minio()

elapsed = time.time() - start_ts
print(
    f"[dynamic] done | mode={mode} rows={len(train_rows)} JS={JS:.4f} "
    f"| loss={best_loss:.6f} | acc@0.15 {base_acc:.2f}%→{new_acc:.2f}% "
    f"| elapsed={elapsed:.2f}s"
)