#!/usr/bin/env python3
"""
ml.train_offline – baseline 训练 + 产出基准预测 + 指标

新增功能：
· 设置 TRIGGER = 1 表示执行训练
· 设置 TRIGGER = 0 表示跳过训练，直接从 MinIO 拉模型
· 结尾写出 Kubeflow V2 必需的 output_metadata.json
"""

# -------------------- 触发器常量 --------------------
TRIGGER = 0   # 改为 0 表示跳过训练，复用 MinIO 中已有 artefacts
# ---------------------------------------------------

import os, sys, datetime, io, time, json
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from shared.minio_helper import (
    s3, BUCKET, load_csv, load_np, save_np, save_bytes
)
from shared.utils         import save_json
from shared.config        import DATA_DIR, MODEL_DIR, RESULT_DIR, TARGET_COL
from shared.features      import FEATURE_COLS
from shared.utils         import calculate_accuracy_within_threshold
from shared.metric_logger import log_metric

# ========== 如果 TRIGGER 为 0，直接复用历史模型 ==========
if TRIGGER == 0:
    print("[offline] TRIGGER = 0 → 跳过训练，复用 MinIO 中的模型和预测结果")
    os.makedirs("/mnt/pvc/models", exist_ok=True)
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

    for fname in ("model.pt", "scaler.pkl"):
        raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{fname}")["Body"].read()
        with open(f"/mnt/pvc/models/{fname}", "wb") as f:
            f.write(raw)
        print(f"[offline] 已下载 {fname}")

    for fname in ("bridge_true.npy", "bridge_pred.npy"):
        try:
            arr = load_np(f"{RESULT_DIR}/{fname}")
            np.save(f"/mnt/pvc/{RESULT_DIR}/{fname}", arr)
            print(f"[offline] 已下载 {fname}")
        except Exception as e:
            print(f"[offline] ❗ 跳过 {fname}：{e}")

    log_metric(component="offline", event="skip_train")
    print("[offline] 跳过训练流程完毕。")
else:
    # ========== 如果 TRIGGER 为 1，正常执行训练 ==========
    SEED = 40
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    class MLPBaseline(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 64),  nn.ReLU(),
                nn.Linear(64,  32),  nn.ReLU(),
                nn.Linear(32,   1),
            )
        def forward(self, x): return self.net(x).squeeze(1)

    # ---------- 读取数据 ----------
    df   = load_csv(f"{DATA_DIR}/combined.csv")
    Xraw = df[FEATURE_COLS].astype(np.float32).values
    y    = df[TARGET_COL].astype(np.float32).values

    # ---------- 训练 ----------
    scaler = StandardScaler().fit(Xraw)
    Xs     = scaler.transform(Xraw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MLPBaseline(Xs.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.SmoothL1Loss()

    Xts = torch.from_numpy(Xs).to(device)
    yts = torch.from_numpy(y).to(device)

    t0 = time.perf_counter()
    for _ in range(200):
        model.train(); opt.zero_grad()
        loss = lossfn(model(Xts), yts); loss.backward(); opt.step()
    train_time_s = round(time.perf_counter() - t0, 3)

    # ---------- 预测 & 评估 ----------
    model.eval()
    with torch.no_grad():
        pred = model(Xts).cpu().numpy().astype(np.float32)

    mae  = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    acc  = calculate_accuracy_within_threshold(y, pred, 0.15)

    print(f"[offline] acc={acc:.2f}% | MAE={mae:.3f} | RMSE={rmse:.3f} | rows={len(y)}")

    # ---------- 保存 artefacts ----------
    os.makedirs("/mnt/pvc/models", exist_ok=True)
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

    joblib.dump(scaler, "/mnt/pvc/models/scaler.pkl")
    torch.save(model.state_dict(), "/mnt/pvc/models/model.pt")

    np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy", y)
    np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy", pred)

    # ---------- 上传到 MinIO ----------
    save_np(f"{RESULT_DIR}/bridge_true.npy", y)
    save_np(f"{RESULT_DIR}/bridge_pred.npy", pred)
    save_bytes(f"{MODEL_DIR}/scaler.pkl", open("/mnt/pvc/models/scaler.pkl", "rb").read())
    save_bytes(f"{MODEL_DIR}/model.pt"  , open("/mnt/pvc/models/model.pt",   "rb").read())

    print("[offline] artefacts pushed to MinIO")

    # ---------- 埋点 ----------
    model_size_mb = round(os.path.getsize("/mnt/pvc/models/model.pt") / 2**20, 3)
    log_metric(
        component="offline",
        event="train_done",
        train_rows=len(y),
        train_time_s=train_time_s,
        model_size_mb=model_size_mb,
        accuracy=round(acc, 2),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
    )

    # ---------- 旧 JSON 输出 ----------
    meta = dict(
        component="offline_train",
        rows=int(len(y)),
        baseline_acc=round(acc, 2),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        utc_end=datetime.datetime.utcnow().isoformat()+"Z",
        model_bytes=os.path.getsize("/mnt/pvc/models/model.pt"),
        train_time_s=train_time_s,
    )
    save_json(f"{RESULT_DIR}/timing/offline_train.json", meta)

    print("[offline] 完成训练并上传 artefacts")

# ========== **新增**：写出 KFP V2 metadata，通知 launcher 继续调度下游 ==========
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json", "w") as f:
    # 如果需要，可以在这里写入 outputs 字段
    json.dump({}, f)

