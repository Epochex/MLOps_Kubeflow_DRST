#!/usr/bin/env python3
"""
ml.dynamic_retrain  JS-aware 网格搜索
------------------------------------------------------------
python -m ml.dynamic_retrain  <js_val>
  • 读取 /mnt/pvc/latest_batch.npy
  • 根据 js_val 选择 param_grid (A/B/C)
  • 枚举超参 → 早停 → 选最优 → 保存到 MinIO/models/model.pt
"""
import sys, io, pathlib, numpy as np, joblib, torch
from itertools import product
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from shared.minio_helper import s3, BUCKET, save_bytes
from shared.config import MODEL_DIR, TARGET_COL
from shared.features import FEATURE_COLS
from ml.model import build_model
from shared.metric_logger import log_metric


JS = float(sys.argv[1])

param_grid_A = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size":    [16, 32],
    "hidden_layers": [(64, 32)],
    "activation":    ["relu"],
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
    "hidden_layers": [(256,128,64), (128,128,64,32)],
    "activation":    ["relu", "gelu"],
}

def grid(js):
    if js <= .05:   return None
    if js <= .15:   return param_grid_A
    if js <= .30:   return param_grid_B
    return param_grid_C

grid_cfg = grid(JS)
if grid_cfg is None:
    print("[dynamic] JS≤0.05 – skip retrain"); sys.exit(0)

# 载 scaler
buf = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read())
scaler = joblib.load(buf)

# 载 latest_batch
batch = np.load("/mnt/pvc/latest_batch.npy", allow_pickle=True)
Xr   = np.array([[row["features"][c] for c in FEATURE_COLS] for row in batch], np.float32)
yr   = np.array([row[TARGET_COL] for row in batch], np.float32)

Xr   = scaler.transform(Xr)
Xtr, Xval, ytr, yval = train_test_split(Xr, yr, test_size=.2, random_state=0)
tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
vl_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))

best_loss, best_state = 1e9, None
device = "cpu"
lossfn = nn.SmoothL1Loss()

for hp in product(*grid_cfg.values()):
    cfg = dict(zip(grid_cfg.keys(), hp))
    model = build_model(cfg, Xtr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=128)
    patience = 0; min_vl = 1e9
    for _ in range(20):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        vl = np.mean([lossfn(model(xb.to(device)), yb.to(device)).item()
                      for xb, yb in vl_dl])
        if vl < min_vl - 1e-4:
            min_vl, patience = vl, 0
        else:
            patience += 1
        if patience >= 3: break

    if min_vl < best_loss:
        best_loss, best_state = min_vl, model.state_dict()
        print(f"[dynamic] ★ new best {best_loss:.4f}  cfg={cfg}")

# 保存
if best_state is not None:
    out = io.BytesIO(); torch.save(best_state, out); out.seek(0)
    save_bytes(f"{MODEL_DIR}/model.pt", out.read())
    print(f"[dynamic] ✅ pushed model.pt  | JS={JS:.3f}  loss={best_loss:.4f}")
log_metric(component="retrain",
           event="model_update",
           value=round(best_loss, 6))

else:
    print("[dynamic] no update")
