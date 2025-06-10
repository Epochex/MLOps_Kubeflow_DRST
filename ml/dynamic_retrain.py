#!/usr/bin/env python3
import sys, io, itertools, joblib, numpy as np, torch, json
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch.nn as nn

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.features     import FEATURE_COLS
from ml.model            import build_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler     import Timer

# 1) 读取 JS divergence
JS = float(sys.argv[1])

# 2) 定义 A/B/C 参数网格
param_grid_A = {"learning_rate":[1e-3,5e-4,1e-4], "batch_size":[16,32]}
param_grid_B = {
    "learning_rate":[1e-3,5e-4], "batch_size":[16,32],
    "hidden_layers":[(64,32),(128,64,32)], "activation":["relu","tanh"]
}
param_grid_C = {
    "learning_rate":[1e-2,1e-3], "batch_size":[16,32,64],
    "hidden_layers":[(256,128,64),(128,128,64,32)],
    "activation":["relu","gelu"], "loss":["Huber","mse"]
}

def grid(js_val):
    if js_val <= JS_SEV1_THRESH/3: return None
    if js_val <= JS_SEV1_THRESH: return param_grid_A
    if js_val <= JS_SEV2_THRESH: return param_grid_B
    return param_grid_C

def severity_tag(js_val):
    if js_val <= JS_SEV1_THRESH: return "Severity-1 Handler"
    if js_val <= JS_SEV2_THRESH: return "Severity-2 Handler"
    return "Severity-K Handler"

grid_cfg = grid(JS)
if grid_cfg is None:
    print("[dynamic] JS below skip threshold → exit")
    sys.exit(0)
sev_name = severity_tag(JS)

# 3) 加载 scaler
buf = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

# 4) 加载最新 batch
batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)
Xr    = np.array([[r["features"][c] for c in FEATURE_COLS] for r in batch], np.float32)
yr    = np.array([r["label"] for r in batch], np.float32)  # ⚠️ 取 ["label"] 而非 TARGET_COL

Xr = scaler.transform(Xr)
Xtr, Xval, ytr, yval = train_test_split(Xr, yr, test_size=0.2, random_state=0)

# 5) TensorDataset
device  = "cpu"
Xtr_t   = torch.from_numpy(Xtr).to(device)
ytr_t   = torch.from_numpy(ytr).to(device)
Xval_t  = torch.from_numpy(Xval).to(device)
yval_t  = torch.from_numpy(yval).to(device)

LOSS_MAP = {"Huber": nn.SmoothL1Loss, "mse": nn.MSELoss}
best_loss, best_state, best_cfg = float("inf"), None, None

# 6) 用 Timer 计整体 Handler 耗时
with Timer(sev_name, "retrain"):
    for hp in itertools.product(*grid_cfg.values()):
        cfg   = dict(zip(grid_cfg.keys(), hp))
        model = build_model(cfg, Xtr.shape[1]).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        lossfn= LOSS_MAP.get(cfg.get("loss","Huber"), nn.SmoothL1Loss)()

        # 训练+验证都包在这个 Timer
        with Timer(f"{sev_name}_config", "retrain"):
            model.train()
            for _ in range(10):
                opt.zero_grad()
                loss = lossfn(model(Xtr_t), ytr_t)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = lossfn(model(Xval_t), yval_t).item()

        if val_loss < best_loss:
            best_loss, best_state, best_cfg = val_loss, model.state_dict(), cfg
            print(f"[dynamic] ★ new best {best_loss:.4f} cfg={cfg}")

# 7) Model Update
with Timer("Model_Update", "retrain"):
    # 保存 state_dict
    buf = io.BytesIO()
    torch.save(best_state, buf); buf.seek(0)
    save_bytes(f"{MODEL_DIR}/model.pt", buf.read())
    # 保存超参配置
    cfg_bytes = json.dumps(best_cfg).encode()
    save_bytes(f"{MODEL_DIR}/last_model_config.json", cfg_bytes, "application/json")
    # 更新时间戳
    now = datetime.utcnow().isoformat() + "Z"
    save_bytes(f"{MODEL_DIR}/last_update_utc.txt", now.encode())

log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update", value=round(best_loss,6))

# 8) 同步所有 retrain 阶段指标到 MinIO
sync_all_metrics_to_minio()
print(f"[dynamic] done, best_loss={best_loss:.6f}, best_cfg={best_cfg}")
