#!/usr/bin/env python3
"""
ml.dynamic_retrain  –  JS-aware 超参搜索 & 在线模型更新
------------------------------------------------------------
调用方式：
    python -m ml.dynamic_retrain <js_val_float>
"""
import sys
import io
import pathlib
import itertools
import joblib
import numpy as np
import torch
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, TARGET_COL
from shared.features     import FEATURE_COLS
from ml.model            import build_model
from shared.metric_logger import log_metric
from shared.profiler     import Timer

# ─────────────────────────────────────────────────────────────
JS = float(sys.argv[1])

# ---------- 参数网格 ----------
param_grid_A = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size":    [16, 32],
}
param_grid_B = {
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [16, 32],
    "hidden_layers": [(64, 32), (128, 64, 32)],
    "activation":    ["relu", "tanh"],
}
param_grid_C = {
    "learning_rate": [1e-2, 1e-3],
    "batch_size":    [16, 32, 64],
    "hidden_layers": [(256, 128, 64), (128, 128, 64, 32)],
    "activation":    ["relu", "gelu"],
    "loss":          ["Huber", "mse"],
}

def grid(js_val: float):
    if js_val <= 0.05:
        return None
    if js_val <= 0.15:
        return param_grid_A
    if js_val <= 0.30:
        return param_grid_B
    return param_grid_C

grid_cfg = grid(JS)
if grid_cfg is None:
    print("[dynamic] JS ≤ 0.05 – skip retrain")
    sys.exit(0)

# ---------- 准备数据 ----------
# 下载并加载 scaler
buf = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

# 加载最新 batch
batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)
Xr = np.array([[row["features"][c] for c in FEATURE_COLS] for row in batch], np.float32)
yr = np.array([row[TARGET_COL] for row in batch], np.float32)

# 划分训练/验证集
Xr = scaler.transform(Xr)
Xtr, Xval, ytr, yval = train_test_split(Xr, yr, test_size=0.2, random_state=0)
tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
vl_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))

device = "cpu"
_LOSS = {
    "Huber": nn.SmoothL1Loss,
    "mse":   nn.MSELoss,
}

def _severity_tag(js_val: float) -> str:
    return ("Severity-1 Handler" if js_val <= 0.3 else
            "Severity-2 Handler" if js_val <= 0.5 else
            "Severity-K Handler")

best_loss, best_state = float("inf"), None

# ---------- 记录重训开始时刻 ----------
start_all = time.perf_counter()

# ---------- 网格搜索 ----------
for hp in itertools.product(*grid_cfg.values()):
    cfg = dict(zip(grid_cfg.keys(), hp))
    model = build_model(cfg, Xtr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    lossfn = _LOSS.get(cfg.get("loss", "Huber"), nn.SmoothL1Loss)()

    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=128)

    # 用 Timer 记录“单次配置”训练+验证的 runtime_ms 与 cpu_time_ms
    with Timer(_severity_tag(JS), "retrain"):
        patience, min_vl = 0, float("inf")
        for _ in range(20):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = lossfn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # 验证集 loss
            vl = np.mean([
                lossfn(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in vl_dl
            ])
            if vl < min_vl - 1e-4:
                min_vl, patience = vl, 0
            else:
                patience += 1
            if patience >= 3:
                break  # early stop

    # 保存最优
    if min_vl < best_loss:
        best_loss, best_state = min_vl, model.state_dict()
        print(f"[dynamic] ★ new best {best_loss:.4f}  cfg={cfg}")

# ---------- 记录重训总耗时并埋点 ----------
runtime_all = round(time.perf_counter() - start_all, 3)
log_metric(
    component="retrain",
    event="retrain_runtime",
    retrain_runtime_s=runtime_all
)

# ---------- 推送最优模型 ----------
if best_state is not None:
    out = io.BytesIO()
    torch.save(best_state, out)
    out.seek(0)
    save_bytes(f"{MODEL_DIR}/model.pt", out.read())

    # 写入推送时点，供消费者计算 deploy_delay_s
    now_iso = datetime.utcnow().isoformat() + "Z"
    save_bytes(f"{MODEL_DIR}/last_update_utc.txt", now_iso.encode(), "text/plain")

    # 模型推送埋点
    log_metric(
        component="retrain",
        event="model_pushed"
    )
    print(f"[dynamic] ✅ pushed model.pt  | JS={JS:.3f}  loss={best_loss:.4f}")

# ---------- 最终更新指标 ----------
log_metric(
    component="retrain",
    event="model_update",
    value=round(best_loss, 6)
)
