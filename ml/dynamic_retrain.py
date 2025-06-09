#!/usr/bin/env python3
import sys, io, itertools, joblib, numpy as np, torch, time
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config       import MODEL_DIR, TARGET_COL, JS_SEV1_THRESH, JS_SEV2_THRESH
from shared.features     import FEATURE_COLS
from ml.model            import build_model
from shared.metric_logger import log_metric
from shared.profiler     import Timer

JS = float(sys.argv[1])

# 定义三套网格 A/B/C（同原）
param_grid_A = { "learning_rate":[1e-3,5e-4,1e-4], "batch_size":[16,32] }
param_grid_B = { "learning_rate":[1e-3,5e-4],   "batch_size":[16,32],
                 "hidden_layers":[(64,32),(128,64,32)], "activation":["relu","tanh"] }
param_grid_C = { "learning_rate":[1e-2,1e-3], "batch_size":[16,32,64],
                 "hidden_layers":[(256,128,64),(128,128,64,32)],
                 "activation":["relu","gelu"], "loss":["Huber","mse"] }

def grid(js_val: float):
    if js_val <= JS_SEV1_THRESH / 3:      
        return None
    if js_val <= JS_SEV1_THRESH:           
        return param_grid_A
    if js_val <= JS_SEV2_THRESH:           
        return param_grid_B
    return param_grid_C                   
def severity_tag(js_val: float) -> str:
    if js_val <= JS_SEV1_THRESH:
        return "Severity-1 Handler"
    if js_val <= JS_SEV2_THRESH:
        return "Severity-2 Handler"
    return "Severity-K Handler"

grid_cfg = grid(JS)
if grid_cfg is None:
    print("[dynamic] JS ≤ skip threshold – skip retrain")
    sys.exit(0)

sev_name = severity_tag(JS)

# 准备数据
buf = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)
Xr = np.array([[row["features"][c] for c in FEATURE_COLS] for row in batch], np.float32)
yr = np.array([row[TARGET_COL] for row in batch], np.float32)
Xr = scaler.transform(Xr)
Xtr, Xval, ytr, yval = train_test_split(Xr, yr, test_size=0.2, random_state=0)
tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
vl_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))

device = "cpu"
_LOSS = {"Huber": nn.SmoothL1Loss, "mse": nn.MSELoss}

best_loss, best_state = float("inf"), None

# —— Adaptation: Severity-X Handler 总耗时 ——  
# 用一个 Timer 统计整个 Handler 的耗时
with Timer(sev_name, "retrain"):
    best_loss, best_state = float("inf"), None
    for hp in itertools.product(*grid_cfg.values()):
        cfg = dict(zip(grid_cfg.keys(), hp))
        model  = build_model(cfg, Xtr.shape[1]).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        lossfn = _LOSS.get(cfg.get("loss","Huber"), nn.SmoothL1Loss)()

        # 对每个配置再用一个 Timer（可选）
        with Timer(f"{sev_name}_config", "retrain"):
            # …训练+验证略…

        if val_loss < best_loss:
            best_loss, best_state = val_loss, model.state_dict()
            print(f"[dynamic] ★ new best {best_loss:.4f} cfg={cfg}")

# Model Update 耗时
with Timer("Model_Update", "retrain"):
    out = io.BytesIO(); torch.save(best_state, out); out.seek(0)
    save_bytes(f"{MODEL_DIR}/model.pt", out.read())
    now = datetime.utcnow().isoformat() + "Z"
    save_bytes(f"{MODEL_DIR}/last_update_utc.txt", now.encode())

log_metric(component="retrain", event="model_pushed")
log_metric(component="retrain", event="model_update", value=round(best_loss,6))